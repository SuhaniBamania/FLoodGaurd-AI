from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from backend.operations.allocator import run_module5
from backend.operations.shelter_allocator import run_module7
from backend.operations.relief_planner import run_module8
from backend.operations.adapter import priority_result_to_incident_row

import cv2
import numpy as np
import json
import os
import subprocess
import math
import time
import urllib.request
import urllib.error
import urllib.parse
import re
from datetime import datetime, timezone

from dotenv import load_dotenv

from backend.services.detector import DroneDetector
from backend.supabase_client import supabase
from backend.operations.db_client import supabase as operations_supabase
from pydantic import BaseModel
from typing import Optional, Dict, Any

from backend.services.flood_risk import predict_flood_risk
from backend.services.sos_extractor import extract_sos_llm
from backend.services.priority import calculate_priority

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SWY_PATH = os.getenv(
    "SWYTCHCODE_PATH",
    r"C:\Users\Admin\AppData\Roaming\npm\swy.cmd"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="FloodGuard Emergency Response API",
    description=(
        "FloodGuard API for drone detection, emergency "
        "shelter discovery, safe routing and AI recommendation."
    ),
    version="2.1.0"
)

# ============================================================
# LOCAL SOS STORAGE
# ============================================================
# SOS is kept locally so it does not depend on the Supabase
# sos_requests table. Supabase remains available for the
# drone feature.
# ============================================================
# LOCAL PERSISTENT SOS STORAGE
# ============================================================

SOS_FILE = "sos_data.json"


def load_sos_store():
    """Load SOS requests from local JSON storage."""
    if not os.path.exists(SOS_FILE):
        return []

    try:
        with open(SOS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, OSError):
        return []


def save_sos_store(store):
    """Save SOS requests to local JSON storage."""
    with open(SOS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            store,
            f,
            indent=2,
            ensure_ascii=False
        )


SOS_STORE = load_sos_store()

SOS_COUNTER = (
    max(
        [
            int(item.get("id", 0))
            for item in SOS_STORE
            if str(item.get("id", "")).isdigit()
        ]
        or [0]
    )
    + 1
)

#
# ============================================================
# FEATURE 1 / 3 / 4 REQUEST MODELS
# ============================================================

class FloodRiskRequest(BaseModel):
    water_level_cm: float
    water_level_rate_cm_per_min: float
    rainfall_mm_per_hr: float
    soil_moisture_pct: float
    elevation_m: float


class SOSRequest(BaseModel):
    message: str


class PriorityRequest(BaseModel):
    sos_data: Dict[str, Any]
    feature2_people_count: Optional[int] = None
    flood_severity: float = 0.0

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
   allow_origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DRONE MODEL
# ============================================================

MODEL_PATH = "backend/models/floodguard_person_v2.pt"

# Detection configuration
# 0.50 removes very weak false-positive predictions while
# keeping reasonably confident person detections.
DETECTION_CONFIDENCE = 0.50

detector = DroneDetector(
    model_path=MODEL_PATH,
    tile_size=512,
    overlap=0.25,
    conf_threshold=DETECTION_CONFIDENCE,
    nms_iou=0.45
)

print("=" * 60)
print("FloodGuard AI detector configuration")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Confidence threshold: {DETECTION_CONFIDENCE}")
print("Tile size: 512")
print("Overlap: 0.25")
print("NMS IoU: 0.45")
print("=" * 60)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "FloodGuard Emergency Response API is running",
        "status": "healthy",
        "version": "2.1.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "drone_model": "floodguard_person_v2.pt",
        "google_maps_configured": bool(GOOGLE_MAPS_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "swytchcode_path": SWY_PATH
    }


# ============================================================
# EXTRACT LOCATION FROM DRONE IMAGE FILENAME
# ============================================================

def _extract_location_from_filename(filename: str):
    """
    Extract location metadata from the uploaded drone image filename.

    Expected filename format:
        location_state_latitude_longitude_flood.jpg

    Examples:
        supaul_bihar_26.1153_86.5951_flood.jpg
        patna_bihar_25.1000_85.1000_flood.jpg
        dibrugarh_assam_27.4845_94.9019_flood.jpg
        malda_west_bengal_25.1372_88.0899_flood.jpg

    The filename is now the source of truth for the drone image
    location. Any live/browser location sent by the frontend is ignored.
    """

    if not filename:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded image must have a location-based filename "
                "in the format: location_state_latitude_longitude_flood.jpg"
            )
        )

    safe_filename = os.path.basename(filename)

    # Remove the extension before parsing.
    stem = os.path.splitext(safe_filename)[0].strip()

    # Match the latitude and longitude from the end of the filename.
    # This also allows negative coordinates.
    pattern = re.compile(
        r"^(?P<location>.+?)_"
        r"(?P<latitude>-?\d+(?:\.\d+)?)_"
        r"(?P<longitude>-?\d+(?:\.\d+)?)_"
        r"flood(?:_[^_]*)?$",
        re.IGNORECASE
    )

    match = pattern.match(stem)

    if not match:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid drone image filename. Expected format: "
                "location_state_latitude_longitude_flood.jpg"
            )
        )

    location_part = match.group("location")

    try:
        parsed_latitude = float(
            match.group("latitude")
        )
        parsed_longitude = float(
            match.group("longitude")
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Latitude or longitude in the filename is invalid."
        )

    if not -90 <= parsed_latitude <= 90:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid latitude in filename: {parsed_latitude}"
        )

    if not -180 <= parsed_longitude <= 180:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid longitude in filename: {parsed_longitude}"
        )

    # Convert the machine-readable filename part into a human-readable
    # location label while preserving the original filename separately.
    location_name = location_part.replace("_", " ").strip()
    location_name = " ".join(location_name.split()).title()

    return {
        "location": location_name,
        "latitude": parsed_latitude,
        "longitude": parsed_longitude,
        "location_part": location_part,
        "source": "filename"
    }


def _extract_explicit_coordinates(message: str):
    if not message:
        return None

    patterns = [
        r"(?i)\blocation\s*[:=-]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
        r"(?i)\bcoordinates?\s*[:=-]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
        r"(?<!\d)(-?\d{1,2}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)(?!\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, message)

        if not match:
            continue

        try:
            latitude = float(match.group(1))
            longitude = float(match.group(2))
        except (TypeError, ValueError):
            continue

        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            return {
                "latitude": latitude,
                "longitude": longitude,
            }

    return None


# ============================================================
# DRONE PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    # These fields are kept for frontend backward compatibility.
    # They are intentionally NOT used for the drone location anymore.
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    location: Optional[str] = Form(None)
):

    # --------------------------------------------------------
    # LOCATION SOURCE
    # --------------------------------------------------------
    # IMPORTANT:
    # The drone image filename is now the source of truth.
    # Any live/browser GPS values sent by the frontend are ignored.
    # --------------------------------------------------------

    filename_metadata = _extract_location_from_filename(
        file.filename
    )

    image_latitude = filename_metadata["latitude"]
    image_longitude = filename_metadata["longitude"]
    image_location = filename_metadata["location"]

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not file.content_type or not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not process the uploaded image."
        )

    # --------------------------------------------------------
    # Run AI detection
    # --------------------------------------------------------

    try:

        detections, annotated_image = detector.predict(
            image
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Format detections
    # --------------------------------------------------------

    formatted_detections = []

    for detection in detections:

        x1, y1, x2, y2, confidence = detection

        formatted_detections.append({

            "x1": round(x1, 2),

            "y1": round(y1, 2),

            "x2": round(x2, 2),

            "y2": round(y2, 2),

            "confidence": round(
                confidence,
                4
            )
        })

    # --------------------------------------------------------
    # Calculate detection statistics
    # --------------------------------------------------------

    people_detected = len(
        formatted_detections
    )

    if people_detected > 0:

        average_confidence = (
            sum(
                detection["confidence"]
                for detection
                in formatted_detections
            )
            /
            people_detected
        )

    else:

        average_confidence = 0.0

    # --------------------------------------------------------
    # Upload original image to Supabase Storage
    # --------------------------------------------------------

    try:

        original_filename = (
            file.filename
            or
            "drone_image.jpg"
        )

        # Prevent directory traversal
        safe_filename = os.path.basename(
            original_filename
        )

        # Make filename unique
        timestamp = int(time.time() * 1000)

        image_path = (
            f"detections/"
            f"{timestamp}_"
            f"{safe_filename}"
        )

        supabase.storage.from_(
            "drone-images"
        ).upload(
            image_path,
            image_bytes,
            {
                "content-type":
                    file.content_type
                    or
                    "image/jpeg"
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Image upload to Supabase failed: "
                f"{str(e)}"
            )
        )

    # --------------------------------------------------------
    # Encode annotated image for existing frontend
    # --------------------------------------------------------

    success, encoded_image = cv2.imencode(
        ".jpg",
        annotated_image
    )

    if not success:

        raise HTTPException(
            status_code=500,
            detail="Could not encode annotated image."
        )

    # --------------------------------------------------------
    # Upload processed/annotated image to Supabase
    # --------------------------------------------------------

    try:

        processed_image_path = (
            f"detections/processed_"
            f"{timestamp}_"
            f"{safe_filename}"
        )

        supabase.storage.from_(
            "drone-images"
        ).upload(
            processed_image_path,
            encoded_image.tobytes(),
            {
                "content-type":
                    "image/jpeg"
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Processed image upload to Supabase failed: "
                f"{str(e)}"
            )
        )

    # --------------------------------------------------------
    # Save detection information to Supabase
    # --------------------------------------------------------

    try:

        detection_record = {

            "image_path":
                processed_image_path,

            "location":
                image_location,

            "latitude":
                image_latitude,

            "longitude":
                image_longitude,

            "location_source":
                "filename",

            # Current detector does not calculate
            # analysis duration.
            "time_span":
                None,

            "people_detected":
                people_detected,

            # Current person detector does not
            # calculate flooded area.
            "flood_area":
                None,

            # Do not use HIGH based only on
            # people detection.
            "severity":
                None,

            "confidence":
                round(
                    average_confidence,
                    4
                ),

            "model_name":
                "floodguard_person_v2.pt",

            "analysis_status":
                "COMPLETED"
        }

        db_response = (
            supabase
            .table("drone_detections")
            .insert(
                detection_record
            )
            .execute()
        )

    except Exception as e:

        # ----------------------------------------------------
        # If DB insert fails, remove the uploaded image
        # so we don't leave an orphaned Storage object.
        # ----------------------------------------------------

        try:

            supabase.storage.from_(
                "drone-images"
            ).remove([
                image_path,
                processed_image_path
            ])

        except Exception:

            pass

        raise HTTPException(
            status_code=500,
            detail=(
                f"Detection data could not be saved "
                f"to Supabase: {str(e)}"
            )
        )

    # --------------------------------------------------------
    # Existing frontend metadata
    # --------------------------------------------------------

    metadata = {

        "success":
            True,

        "filename":
            file.filename,

        "people_count":
            people_detected,

        "detections":
            formatted_detections,

        # New Supabase-related information
        "latitude":
            image_latitude,

        "longitude":
            image_longitude,

        "location":
            image_location,

        "location_source":
            "filename",

        "image_path":
            processed_image_path,

        "confidence":
            round(
                average_confidence,
                4
            ),

        "model_name":
            "floodguard_person_v2.pt",

        "analysis_status":
            "COMPLETED"
    }

    metadata_json = json.dumps(
        metadata
    )

    # --------------------------------------------------------
    # Keep existing multipart/mixed response
    # --------------------------------------------------------

    boundary = "FloodGuardBoundary"

    body = (

        f"--{boundary}\r\n"

        f"Content-Type: application/json\r\n"

        f"Content-Disposition: "
        f"form-data; name=\"metadata\"\r\n"

        f"\r\n"

        f"{metadata_json}\r\n"

        f"--{boundary}\r\n"

        f"Content-Type: image/jpeg\r\n"

        f"Content-Disposition: inline; "
        f"name=\"image\"; filename=\"result.jpg\"\r\n"

        f"\r\n"

    ).encode("utf-8")

    body += encoded_image.tobytes()

    body += (
        f"\r\n--{boundary}--\r\n"
    ).encode("utf-8")

    return Response(
        content=body,
        media_type=(
            f"multipart/mixed; "
            f"boundary={boundary}"
        )
    )
# ============================================================
# FEATURE 1 — FLOOD RISK
# ============================================================

@app.post("/flood-risk")
async def flood_risk(request: FloodRiskRequest):

    try:

        result = predict_flood_risk(
            water_level_cm=request.water_level_cm,
            water_level_rate_cm_per_min=(
                request.water_level_rate_cm_per_min
            ),
            rainfall_mm_per_hr=request.rainfall_mm_per_hr,
            soil_moisture_pct=request.soil_moisture_pct,
            elevation_m=request.elevation_m
        )

        risk_to_severity = {
            "LOW": 0.25,
            "MEDIUM": 0.50,
            "HIGH": 0.75,
            "CRITICAL": 1.00
        }

        result["flood_severity"] = risk_to_severity.get(
            result["risk_label"],
            0.0
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Flood risk prediction failed: {str(e)}"
        )
    
    # ============================================================
# FEATURE 3 — SOS INTELLIGENCE
# ============================================================


# ============================================================
# GEOCODE SOS LOCATION
# ============================================================

def _geocode_sos_location(extracted):
    """
    Add coordinates from the SOS human-readable location.

    Primary:
        Google Geocoding API

    Fallback:
        Google Places Text Search API

    No browser/device GPS is used.
    """
    if not isinstance(extracted, dict):
        return False

    location_data = extracted.get("location")

    if isinstance(location_data, str):
        location_data = {"text": location_data.strip()}
    elif isinstance(location_data, dict):
        location_data = dict(location_data)
    else:
        location_data = {}

    location_text = str(
        location_data.get("text")
        or location_data.get("name")
        or ""
    ).strip()

    if not location_text or location_text.lower() in {
        "location not provided",
        "location not available",
        "not provided",
        "unknown",
        "none",
        "null",
    }:
        extracted["location"] = location_data
        return False

    # Do not overwrite coordinates that already exist.
    if (
        location_data.get("latitude") is not None
        and location_data.get("longitude") is not None
    ):
        extracted["location"] = location_data
        return False

    if not GOOGLE_MAPS_API_KEY:
        print(
            "⚠️ GOOGLE_MAPS_API_KEY not configured; "
            "SOS coordinates unavailable."
        )
        extracted["location"] = location_data
        return False

    address = location_text
    if "india" not in address.lower():
        address = f"{address}, India"

    # ============================================================
    # 1. GOOGLE GEOCODING API
    # ============================================================
    geocode_url = (
        "https://maps.googleapis.com/maps/api/geocode/json?"
        + urllib.parse.urlencode({
            "address": address,
            "key": GOOGLE_MAPS_API_KEY,
        })
    )

    try:
        with urllib.request.urlopen(
            geocode_url,
            timeout=10
        ) as response:
            data = json.loads(
                response.read().decode(
                    "utf-8",
                    errors="replace"
                )
            )

        results = data.get("results", [])
        api_status = data.get("status")

        if results:
            coordinates = results[0].get(
                "geometry",
                {}
            ).get(
                "location",
                {}
            )

            latitude = coordinates.get("lat")
            longitude = coordinates.get("lng")

            if latitude is not None and longitude is not None:
                location_data["latitude"] = float(latitude)
                location_data["longitude"] = float(longitude)
                location_data["geocoded"] = True
                location_data["geocoded_address"] = (
                    results[0].get("formatted_address")
                )
                location_data["coordinate_source"] = "geocoding"

                extracted["location"] = location_data

                print(
                    "📍 SOS location geocoded:",
                    location_text,
                    "→",
                    latitude,
                    longitude
                )
                return True

        # IMPORTANT: show the real Google response status.
        print(
            f"⚠️ Google Geocoding returned no usable result for "
            f"'{location_text}'. Status: {api_status}. "
            f"Error: {data.get('error_message', 'none')}"
        )

    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode(
                "utf-8",
                errors="replace"
            )
        except Exception:
            error_body = ""

        print(
            f"⚠️ Google Geocoding HTTP error for "
            f"'{location_text}': {e.code} {error_body[:500]}"
        )

    except Exception as e:
        print(
            f"⚠️ Google Geocoding failed for "
            f"'{location_text}': {e}"
        )

    # ============================================================
    # 2. GOOGLE PLACES TEXT SEARCH FALLBACK
    # ============================================================
    # Places search is already used by FloodGuard for shelters.
    # This fallback lets us resolve city/district names even when
    # the Geocoding API returns ZERO_RESULTS.
    # ============================================================

    places_url = (
        "https://places.googleapis.com/v1/places:searchText"
    )

    places_payload = {
        "textQuery": address,
        "maxResultCount": 1,
        "languageCode": "en",
    }

    places_request = urllib.request.Request(
        places_url,
        data=json.dumps(
            places_payload
        ).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
            "X-Goog-FieldMask": (
                "places.displayName,"
                "places.formattedAddress,"
                "places.location"
            ),
        },
    )

    try:
        with urllib.request.urlopen(
            places_request,
            timeout=10
        ) as response:
            places_data = json.loads(
                response.read().decode(
                    "utf-8",
                    errors="replace"
                )
            )

        places = places_data.get("places", [])

        if places:
            place = places[0]
            coordinates = place.get(
                "location",
                {}
            )

            latitude = coordinates.get("latitude")
            longitude = coordinates.get("longitude")

            if latitude is not None and longitude is not None:
                location_data["latitude"] = float(latitude)
                location_data["longitude"] = float(longitude)
                location_data["geocoded"] = True
                location_data["geocoded_address"] = (
                    place.get("formattedAddress")
                )
                location_data["coordinate_source"] = "places"

                extracted["location"] = location_data

                print(
                    "📍 SOS location resolved via Places:",
                    location_text,
                    "→",
                    latitude,
                    longitude
                )
                return True

        print(
            f"⚠️ Google Places returned no usable result for "
            f"'{location_text}'. "
            f"Error: {places_data.get('error', 'none')}"
        )

    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode(
                "utf-8",
                errors="replace"
            )
        except Exception:
            error_body = ""

        print(
            f"⚠️ Google Places HTTP error for "
            f"'{location_text}': {e.code} {error_body[:500]}"
        )

    except Exception as e:
        print(
            f"⚠️ Google Places fallback failed for "
            f"'{location_text}': {e}"
        )

    extracted["location"] = location_data
    return False


@app.post("/sos")
async def analyze_sos(request: SOSRequest):
    global SOS_COUNTER

    try:
        # --------------------------------------------------------
        # FEATURE 3 — SOS INTELLIGENCE
        # --------------------------------------------------------
        result = extract_sos_llm(request.message)
        explicit_coordinates = _extract_explicit_coordinates(
            request.message
        )

        if explicit_coordinates:
            if not isinstance(result.get("location"), dict):
                result["location"] = {}

            result["location"]["latitude"] = (
                explicit_coordinates["latitude"]
            )
            result["location"]["longitude"] = (
                explicit_coordinates["longitude"]
            )

        print(
            "📍 SOS ROUTING LOCATION:",
            result.get("location")
        )

        # Convert the Lyzr location (e.g. Dibrugarh, Assam) to coordinates.
        # No browser/device GPS is used.
        _geocode_sos_location(result)

        # --------------------------------------------------------
        # FEATURE 4 — PRIORITY INTELLIGENCE
        # --------------------------------------------------------
        priority_result = calculate_priority(
            sos_data=result,
            feature2_people_count=None,
            flood_severity=0.0
        )

        location = result.get("location", {})
        priority_result.update({
            "incident_id": f"INC{SOS_COUNTER}",
            "people": priority_result["inputs"]["effective_people"],
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "status": "WAITING_FOR_RESCUE",
        })
        SOS_COUNTER += 1

        result["priority"] = priority_result
        try:
            location_text = result.get("location", {}).get("text")
            incident_row = priority_result_to_incident_row(
                priority_result, location_text=location_text
            )
            incident_columns = {
                "incident_id", "location_text", "latitude", "longitude",
                "people_count", "priority_score", "priority_level", "status",
                "people_remaining"
            }
            operations_supabase.table("rescue_incidents").upsert(
                {
                    key: value
                    for key, value in incident_row.items()
                    if key in incident_columns
                }
            ).execute()
        except Exception as log_err:
            raise RuntimeError(
                f"Failed to save incident to rescue_incidents: {log_err}"
            )

        # Allocate boats immediately using priority score and remaining capacity.
        run_module5()

        incident_id = priority_result["incident_id"]
        assignments = (
            operations_supabase.table("rescue_assignments")
            .select("*")
            .eq("incident_id", incident_id)
            .execute()
            .data
        )
        incident = (
            operations_supabase.table("rescue_incidents")
            .select("*")
            .eq("incident_id", incident_id)
            .single()
            .execute()
            .data
        )

        sos_record = {
            "id": incident_id,
            "created_at": incident["created_at"],
            "status": "ASSIGNED" if assignments else "PENDING",
            "original_message": request.message,
            "extracted_data": result,
            "assignments": assignments,
        }

        SOS_STORE.append(sos_record)
        save_sos_store(SOS_STORE)

        return sos_record

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        print("🔥🔥🔥 SOS ERROR:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"SOS analysis failed: {str(e)}"
        )


# ============================================================
# GET STORED SOS REQUESTS
# ============================================================

@app.get("/sos")
def get_sos_requests():
    try:
        # --------------------------------------------------------
        # READ EXISTING SOS DATA ONLY
        # Do NOT geocode here.
        # Do NOT call external APIs here.
        # --------------------------------------------------------

        incidents = (
            operations_supabase
            .table("rescue_incidents")
            .select("*")
            .order("created_at", desc=True)
            .execute()
            .data
        )

        incident_ids = [
            incident["incident_id"]
            for incident in incidents
            if incident.get("incident_id")
        ]

        assignments = (
            operations_supabase
            .table("rescue_assignments")
            .select("*")
            .in_("incident_id", incident_ids)
            .execute()
            .data
            if incident_ids
            else []
        )

        # --------------------------------------------------------
        # GROUP ASSIGNMENTS BY INCIDENT
        # --------------------------------------------------------

        assignments_by_incident = {}

        for assignment in assignments:
            incident_id = str(
                assignment.get("incident_id")
            )

            assignments_by_incident.setdefault(
                incident_id,
                []
            ).append(assignment)

        # --------------------------------------------------------
        # LOCAL SOS STORE LOOKUP
        # --------------------------------------------------------

        stored_sos = {
            str(item.get("id")): item
            for item in SOS_STORE
        }

        requests = []

        # --------------------------------------------------------
        # BUILD RESPONSE
        # --------------------------------------------------------

        for incident in incidents:

            incident_id = str(
                incident.get("incident_id")
            )

            incident_assignments = (
                assignments_by_incident.get(
                    incident_id,
                    []
                )
            )

            stored_request = stored_sos.get(
                incident_id,
                {}
            )

            extracted_data = stored_request.get(
                "extracted_data",
                {}
            )

            # Backend may have stored JSON as a string
            if isinstance(extracted_data, str):
                try:
                    extracted_data = json.loads(
                        extracted_data
                    )
                except (
                    json.JSONDecodeError,
                    TypeError
                ):
                    extracted_data = {}

            if not isinstance(
                extracted_data,
                dict
            ):
                extracted_data = {}

            # ----------------------------------------------------
            # LOCATION
            # ----------------------------------------------------

            existing_location = (
                extracted_data.get(
                    "location",
                    {}
                )
            )

            if not isinstance(
                existing_location,
                dict
            ):
                existing_location = {}

            extracted_data["location"] = {
                **existing_location,

                "text":
                    incident.get(
                        "location_text"
                    ),

                "latitude":
                    incident.get(
                        "latitude"
                    ),

                "longitude":
                    incident.get(
                        "longitude"
                    ),
            }

            # ----------------------------------------------------
            # PEOPLE
            # ----------------------------------------------------

            existing_people = (
                extracted_data.get(
                    "people",
                    {}
                )
            )

            if not isinstance(
                existing_people,
                dict
            ):
                existing_people = {}

            extracted_data["people"] = {
                **existing_people,

                "total":
                    incident.get(
                        "people_count"
                    ),
            }

            # ----------------------------------------------------
            # PRIORITY
            # ----------------------------------------------------

            existing_priority = (
                extracted_data.get(
                    "priority",
                    {}
                )
            )

            if not isinstance(
                existing_priority,
                dict
            ):
                existing_priority = {}

            extracted_data["priority"] = {
                **existing_priority,

                "priority_score":
                    incident.get(
                        "priority_score"
                    ),

                "priority_level":
                    incident.get(
                        "priority_level"
                    ),
            }

            # ----------------------------------------------------
            # FINAL REQUEST
            # ----------------------------------------------------

            requests.append({

                "id":
                    incident.get(
                        "incident_id"
                    ),

                "created_at":
                    incident.get(
                        "created_at"
                    ),

                "status": (
                    "ASSIGNED"
                    if incident_assignments
                    else "PENDING"
                ),

                "original_message":
                    stored_request.get(
                        "original_message",
                        extracted_data.get(
                            "original_message",
                            ""
                        )
                    ),

                "extracted_data":
                    extracted_data,

                "assignments":
                    incident_assignments,
            })

        return {
            "requests": requests
        }

    except Exception as e:

        print(
            "🔥 GET /sos ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not fetch SOS requests: "
                f"{str(e)}"
            )
        )


# ============================================================
# FEATURE 4 — MULTI-MODAL PRIORITY
# ============================================================

@app.post("/priority")
async def calculate_emergency_priority(
    request: PriorityRequest
):

    try:

        result = calculate_priority(
            sos_data=request.sos_data,
            feature2_people_count=(
                request.feature2_people_count
            ),
            flood_severity=request.flood_severity
        )
        try:
            location_text = request.sos_data.get("location", {}).get("text")
            incident_row = priority_result_to_incident_row(
                result, location_text=location_text
            )
            incident_columns = {
                "incident_id", "location_text", "latitude", "longitude",
                "people_count", "priority_score", "priority_level", "status",
                "people_remaining"
            }
            operations_supabase.table("rescue_incidents").upsert(
                {
                    key: value
                    for key, value in incident_row.items()
                    if key in incident_columns
                }
            ).execute()
        except Exception as log_err:
            print(f"[WARN] Failed to save incident to rescue_incidents: {log_err}")
            
        
        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Priority calculation failed: {str(e)}"
        )


# ============================================================
# FEATURE 4 — MULTI-MODAL PRIORITY DASHBOARD
# ============================================================

# ============================================================
# FEATURE 4 — MULTI-MODAL PRIORITY DASHBOARD
# ============================================================

@app.get("/priority-dashboard")
async def priority_dashboard(
    flood_severity: float = 0.0,
    match_radius_km: float = 5.0
):
    """
    FEATURE 4 — MULTI-MODAL PRIORITY DASHBOARD

    Combines:

        Feature 3:
            SOS intelligence
            - location
            - people
            - vulnerable people
            - needs
            - request type

        Feature 2:
            Drone intelligence
            - people detected
            - drone location
            - drone image

        Feature 1:
            Flood risk severity

    For every active SOS:

        1. Find the nearest drone observation.
        2. Combine SOS + drone + flood information.
        3. Calculate priority using calculate_priority().
        4. Sort all emergencies from highest score to lowest.
        5. Assign an explicit rank.

    Drone location is taken from the location information
    stored when the drone image was processed.
    """

    try:

        # ========================================================
        # 1. VALIDATE PARAMETERS
        # ========================================================

        flood_severity = max(
            0.0,
            min(float(flood_severity), 1.0)
        )

        match_radius_km = max(
            0.1,
            float(match_radius_km)
        )


        # ========================================================
        # 2. GET ACTIVE SOS REQUESTS
        # ========================================================
        #
        # SOS is stored locally in SOS_STORE.
        # We DO NOT use Supabase for SOS.
        #

        active_statuses = {
            "pending",
            "active",
            "open",
            "new"
        }

        sos_records = list(SOS_STORE)

        # Newest SOS first
        sos_records.reverse()

        active_sos = [
            sos
            for sos in sos_records
            if str(
                sos.get("status", "Pending")
            ).lower() in active_statuses
        ]


        # Backfill coordinates here too for the Priority dashboard.
        changed = False
        for sos in active_sos:
            extracted = sos.get("extracted_data")
            if isinstance(extracted, str):
                try:
                    extracted = json.loads(extracted)
                except (json.JSONDecodeError, TypeError):
                    extracted = None
            if isinstance(extracted, dict):
                location = extracted.get("location")
                has_coordinates = (
                    isinstance(location, dict)
                    and location.get("latitude") is not None
                    and location.get("longitude") is not None
                )
                if not has_coordinates and _geocode_sos_location(extracted):
                    sos["extracted_data"] = extracted
                    changed = True
        if changed:
            save_sos_store(SOS_STORE)


        # ========================================================
        # 3. GET DRONE OBSERVATIONS
        # ========================================================
        #
        # Drone data continues to come from Supabase.
        #
        # The drone records contain:
        # - location
        # - latitude
        # - longitude
        # - people_detected
        # - image_path
        # - confidence
        #

        drone_response = (
            supabase
            .table("drone_detections")
            .select("*")
            .eq(
                "analysis_status",
                "COMPLETED"
            )
            .order(
                "id",
                desc=True
            )
            .execute()
        )

        drone_records = (
            getattr(
                drone_response,
                "data",
                None
            )
            or []
        )


        # ========================================================
        # 4. PROCESS EACH SOS
        # ========================================================

        ranked = []

        for sos in active_sos:

            # ====================================================
            # 4A. GET EXTRACTED SOS DATA
            # ====================================================

            raw_extracted = sos.get(
                "extracted_data",
                {}
            )

            if isinstance(
                raw_extracted,
                str
            ):
                try:
                    extracted = json.loads(
                        raw_extracted
                    )

                except (
                    json.JSONDecodeError,
                    TypeError
                ):
                    extracted = {}

            elif isinstance(
                raw_extracted,
                dict
            ):
                extracted = raw_extracted

            else:
                extracted = {}


            # ====================================================
            # 4B. EXTRACT SOS LOCATION
            # ====================================================

            location_data = extracted.get(
                "location",
                {}
            )

            sos_location = ""

            sos_latitude = None

            sos_longitude = None


            if isinstance(
                location_data,
                dict
            ):

                sos_location = str(
                    location_data.get("text")
                    or location_data.get("name")
                    or ""
                ).strip()


                # Latitude
                try:

                    if (
                        location_data.get(
                            "latitude"
                        ) is not None
                    ):

                        sos_latitude = float(
                            location_data.get(
                                "latitude"
                            )
                        )

                except (
                    TypeError,
                    ValueError
                ):

                    sos_latitude = None


                # Longitude
                try:

                    if (
                        location_data.get(
                            "longitude"
                        ) is not None
                    ):

                        sos_longitude = float(
                            location_data.get(
                                "longitude"
                            )
                        )

                except (
                    TypeError,
                    ValueError
                ):

                    sos_longitude = None


            elif isinstance(
                location_data,
                str
            ):

                sos_location = (
                    location_data.strip()
                )


            # ====================================================
            # 4C. MATCH DRONE TO SOS LOCATION
            # ====================================================
            #
            # IMPORTANT:
            # Match by normalized LOCATION NAME first.
            #
            # Why?
            # The SOS LLM can sometimes extract an approximate or
            # incorrect GPS coordinate from natural-language text.
            # The drone filename, however, is our trusted source:
            #
            #   supaul_bihar_26.1153_86.5951_flood.png
            #       -> Supaul Bihar
            #
            # Therefore:
            #   1. Location-name match first
            #   2. GPS distance fallback only if no name match exists
            #
            # This fixes cases such as:
            #   SOS: "Supaul, Bihar"
            #   Drone: "Supaul Bihar"
            # ====================================================

            def _normalize_location_name(value):
                if value is None:
                    return ""

                normalized = str(value).lower().strip()

                # Treat underscores/hyphens/commas as separators.
                normalized = re.sub(
                    r"[_,-]+",
                    " ",
                    normalized
                )

                # Remove common non-location filler words.
                normalized = re.sub(
                    r"\b(near|at|in|around|district|city)\b",
                    " ",
                    normalized
                )

                # Collapse whitespace.
                normalized = re.sub(
                    r"\s+",
                    " ",
                    normalized
                ).strip()

                return normalized

            sos_location_normalized = _normalize_location_name(
                sos_location
            )

            best_drone = None
            best_distance = float("inf")

            # ----------------------------------------------------
            # 4C-1. LOCATION NAME MATCH — PRIMARY
            # ----------------------------------------------------

            if sos_location_normalized:
                for drone in drone_records:

                    drone_location_normalized = (
                        _normalize_location_name(
                            drone.get("location")
                        )
                    )

                    if not drone_location_normalized:
                        continue

                    if (
                        sos_location_normalized
                        == drone_location_normalized
                        or
                        sos_location_normalized
                        in drone_location_normalized
                        or
                        drone_location_normalized
                        in sos_location_normalized
                    ):
                        best_drone = drone
                        best_distance = None

                        # Prefer the newest matching drone record.
                        break

            # ----------------------------------------------------
            # 4C-2. GPS MATCH — FALLBACK
            # ----------------------------------------------------

            if best_drone is None:
                for drone in drone_records:

                    drone_lat = drone.get("latitude")
                    drone_lon = drone.get("longitude")

                    if (
                        sos_latitude is None
                        or sos_longitude is None
                        or drone_lat is None
                        or drone_lon is None
                    ):
                        continue

                    try:
                        distance = calculate_distance_km(
                            sos_latitude,
                            sos_longitude,
                            float(drone_lat),
                            float(drone_lon)
                        )

                        if (
                            distance <= match_radius_km
                            and distance < best_distance
                        ):
                            best_drone = drone
                            best_distance = distance

                    except (TypeError, ValueError):
                        pass
            
            # ====================================================
            # 4C-3. DRONE LOCATION FALLBACK
            # ====================================================
            # Location priority:
            #   1. SOS location coordinates, when available
            #   2. Matched drone coordinates, when SOS coordinates
            #      are missing
            #
            # The drone filename is the trusted source for drone GPS.
            # ====================================================

            if best_drone:
                if sos_latitude is None or sos_longitude is None:
                    drone_lat = best_drone.get("latitude")
                    drone_lon = best_drone.get("longitude")

                    if drone_lat is not None and drone_lon is not None:
                        try:
                            sos_latitude = float(drone_lat)
                            sos_longitude = float(drone_lon)

                            print(
                                "📍 Using DRONE location as SOS fallback:",
                                sos_latitude,
                                sos_longitude
                            )
                        except (TypeError, ValueError):
                            pass

                if not sos_location:
                    sos_location = str(
                        best_drone.get("location") or ""
                    ).strip()


            # ====================================================
            # 4D. GET DRONE PEOPLE COUNT
            # ====================================================

            drone_people = 0


            if best_drone:

                try:

                    drone_people = int(
                        best_drone.get(
                            "people_detected"
                        )
                        or 0
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    drone_people = 0


            # ====================================================
            # 4F. FLOOD SEVERITY — FEATURE 1
            # ====================================================

            effective_flood_severity = (
                flood_severity
            )


            # If Feature 1 severity was not supplied,
            # use severity available in the matched
            # drone record as a fallback.

            if (
                effective_flood_severity
                == 0.0
                and best_drone
            ):

                severity_map = {

                    "LOW": 0.25,

                    "MEDIUM": 0.50,

                    "HIGH": 0.75,

                    "CRITICAL": 1.00
                }


                drone_severity = (
                    best_drone.get(
                        "severity"
                    )
                )


                if drone_severity:

                    effective_flood_severity = (
                        severity_map.get(
                            str(
                                drone_severity
                            ).upper(),
                            0.0
                        )
                    )


            # ====================================================
            # 4G. CALCULATE MULTI-MODAL PRIORITY
            # ====================================================

            priority_result = (
                calculate_priority(

                    sos_data=extracted,

                    feature2_people_count=(
                        drone_people
                        if best_drone
                        else None
                    ),

                    flood_severity=(
                        effective_flood_severity
                    )
                )
            )


            # ====================================================
            # 4H. BUILD COMPLETE EMERGENCY RECORD
            # ====================================================

            emergency = {

                "sos_id": sos.get(
                    "id"
                ),

                "created_at": sos.get(
                    "created_at"
                ),

                "status": sos.get(
                    "status"
                ),


                # -----------------------------------------------
                # LOCATION
                # -----------------------------------------------

                "location": sos_location,

                "latitude": sos_latitude,

                "longitude": sos_longitude,


                # -----------------------------------------------
                # FEATURE 3 — SOS
                # -----------------------------------------------

                "sos_data": extracted,

                "original_message": sos.get(
                    "original_message"
                ),


                # -----------------------------------------------
                # FEATURE 2 — DRONE
                # -----------------------------------------------

                "drone": {

                    "available": (
                        best_drone
                        is not None
                    ),

                    "people_detected": (
                        drone_people
                    ),

                    "location": (
                        best_drone.get(
                            "location"
                        )
                        if best_drone
                        else None
                    ),

                    "latitude": (
                        best_drone.get(
                            "latitude"
                        )
                        if best_drone
                        else None
                    ),

                    "longitude": (
                        best_drone.get(
                            "longitude"
                        )
                        if best_drone
                        else None
                    ),

                    "distance_from_sos_km": (

                        round(
                            best_distance,
                            2
                        )

                        if isinstance(
                            best_distance,
                            (
                                int,
                                float
                            )
                        )

                        and
                        best_distance
                        != float("inf")

                        else None
                    ),

                    "image_path": (

                        best_drone.get(
                            "image_path"
                        )

                        if best_drone
                        else None
                    ),

                    "confidence": (

                        best_drone.get(
                            "confidence"
                        )

                        if best_drone
                        else None
                    ),

                    "model_name": (

                        best_drone.get(
                            "model_name"
                        )

                        if best_drone
                        else None
                    ),

                    "analysis_status": (

                        best_drone.get(
                            "analysis_status"
                        )

                        if best_drone
                        else None
                    )
                },


                # -----------------------------------------------
                # FEATURE 1 — FLOOD
                # -----------------------------------------------

                "flood": {

                    "severity": (
                        effective_flood_severity
                    ),

                    "available": (
                        effective_flood_severity
                        > 0
                    )
                },


                # -----------------------------------------------
                # FINAL PRIORITY
                # -----------------------------------------------

                "priority": priority_result
            }


            ranked.append(
                emergency
            )


        # ========================================================
        # 5. SORT BY ACTUAL PRIORITY SCORE
        # ========================================================
        #
        # IMPORTANT:
        #
        # calculate_priority() returns:
        #
        #     "priority_score"
        #
        # NOT:
        #
        #     "score"
        #
        # So we explicitly sort using priority_score.
        #

        ranked.sort(

            key=lambda item: (

                item
                .get(
                    "priority",
                    {}
                )
                .get(
                    "priority_score",
                    0
                )

            ),

            reverse=True
        )


        # ========================================================
        # 6. ASSIGN RANK
        # ========================================================

        for index, item in enumerate(
            ranked,
            start=1
        ):

            item["rank"] = index


        # ========================================================
        # 7. RETURN FINAL FEATURE 4 RESPONSE
        # ========================================================

        return {

            "success": True,

            "count": len(
                ranked
            ),

            "match_radius_km": (
                match_radius_km
            ),

            "flood_severity_used": (
                flood_severity
            ),

            "emergencies": ranked
        }


    except Exception as e:

        print(
            "🔥 PRIORITY DASHBOARD ERROR:",
            repr(e)
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Priority dashboard failed: "
                f"{str(e)}"
            )
        )

# ============================================================
# DISTANCE HELPER
# ============================================================

def calculate_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
):
    """
    Calculate straight-line distance using Haversine formula.
    """

    earth_radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(
        lat2 - lat1
    )

    delta_lon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1_rad)
        *
        math.cos(lat2_rad)
        *
        math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius_km * c


# ============================================================
# FIND SHELTERS IN THE FLOODGUARD DATABASE
# ============================================================

def _get_shelters_from_floodguard_database(
    latitude: float,
    longitude: float
):
    try:
        shelter_rows = (
            operations_supabase
            .table("shelters")
            .select("*")
            .execute()
            .data
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load shelter data: {str(e)}"
        )

    shelters = []

    for shelter in shelter_rows or []:
        if not isinstance(shelter, dict):
            continue

        try:
            shelter_latitude = float(shelter.get("latitude"))
            shelter_longitude = float(shelter.get("longitude"))
        except (TypeError, ValueError):
            continue

        if (
            not math.isfinite(shelter_latitude)
            or not math.isfinite(shelter_longitude)
            or not -90 <= shelter_latitude <= 90
            or not -180 <= shelter_longitude <= 180
        ):
            continue

        status = str(shelter.get("status") or "").strip().upper()
        if status in {"FULL", "CLOSED", "UNAVAILABLE"}:
            continue

        try:
            capacity_remaining = float(shelter.get("capacity_remaining"))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(capacity_remaining) or capacity_remaining <= 0:
            continue

        distance_km = calculate_distance_km(
            latitude,
            longitude,
            shelter_latitude,
            shelter_longitude
        )

        shelters.append({
            **shelter,
            "id": shelter.get("shelter_id"),
            "name": shelter.get("name") or "Emergency Shelter",
            "address": shelter.get("location_text") or "Address unavailable",
            "latitude": shelter_latitude,
            "longitude": shelter_longitude,
            "distance_km": round(distance_km, 2),
        })

    # --------------------------------------------------------
    # IMPORTANT:
    # Sort by actual distance BEFORE using the shelters.
    # --------------------------------------------------------

    shelters.sort(
        key=lambda item: item.get(
            "distance_km",
            float("inf")
        )
    )

    return shelters


# ============================================================
# SHELTERS ENDPOINT
# ============================================================

@app.get("/shelters")
async def find_shelters(
    latitude: float,
    longitude: float
):

    if not -90 <= latitude <= 90:

        raise HTTPException(
            status_code=400,
            detail="Invalid latitude."
        )

    if not -180 <= longitude <= 180:

        raise HTTPException(
            status_code=400,
            detail="Invalid longitude."
        )

    shelters = _get_shelters_from_floodguard_database(
        latitude,
        longitude
    )

    return {

        "success": True,

        "source": (
            "FloodGuard Supabase shelters table"
        ),

        "origin": {

            "latitude": latitude,

            "longitude": longitude
        },

        "count": len(shelters),

        "shelters": shelters
    }


# ============================================================
# GOOGLE ROUTES API
# ============================================================

def _google_routes_request(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float
):

    if not GOOGLE_MAPS_API_KEY:
        return None

    url = (
        "https://routes.googleapis.com/"
        "directions/v2:computeRoutes"
    )

    payload = {

        "origin": {
            "location": {
                "latLng": {
                    "latitude": float(
                        origin_latitude
                    ),
                    "longitude": float(
                        origin_longitude
                    )
                }
            }
        },

        "destination": {
            "location": {
                "latLng": {
                    "latitude": float(
                        destination_latitude
                    ),
                    "longitude": float(
                        destination_longitude
                    )
                }
            }
        },

        "travelMode": "DRIVE",

        "routingPreference": "TRAFFIC_AWARE",

        "computeAlternativeRoutes": False,

        "languageCode": "en",

        "units": "METRIC"
    }

    request = urllib.request.Request(

        url,

        data=json.dumps(
            payload
        ).encode("utf-8"),

        method="POST",

        headers={

            "Content-Type":
                "application/json",

            "X-Goog-Api-Key":
                GOOGLE_MAPS_API_KEY,

            "X-Goog-FieldMask": (
                "routes.distanceMeters,"
                "routes.duration,"
                "routes.polyline.encodedPolyline"
            )
        }
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            raw = response.read().decode(
                "utf-8",
                errors="replace"
            )

            data = json.loads(
                raw
            )

            routes = data.get(
                "routes",
                []
            )

            if not routes:
                return None

            return {
                "route": routes[0],
                "provider": "Google Routes API"
            }

    except urllib.error.HTTPError as e:

        error_body = e.read().decode(
            "utf-8",
            errors="replace"
        )

        print(
            f"Google Routes API returned {e.code}: "
            f"{error_body[:1000]}"
        )

        return None

    except Exception as e:

        print(
            "Google Routes API failed:",
            str(e)
        )

        return None


# ============================================================
# OSRM FALLBACK ROUTING
# ============================================================

def _osrm_route_request(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float
):
    """
    Free fallback routing service.

    This prevents the Safe Route page from failing completely
    when Google Routes API is unavailable or returns 401.

    OSRM coordinates are longitude,latitude.
    """

    url = (
        "https://router.project-osrm.org/"
        "route/v1/driving/"
        f"{origin_longitude},{origin_latitude};"
        f"{destination_longitude},{destination_latitude}"
        "?overview=full&geometries=geojson"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FloodGuard/1.0"
        }
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            raw = response.read().decode(
                "utf-8",
                errors="replace"
            )

            data = json.loads(
                raw
            )

    except Exception as e:

        print(
            "OSRM routing failed:",
            str(e)
        )

        return None

    routes = data.get(
        "routes",
        []
    )

    if not routes:
        return None

    route = routes[0]

    distance_meters = route.get(
        "distance"
    )

    duration_seconds = route.get(
        "duration"
    )

    geometry = (
        route.get(
            "geometry",
            {}
        )
    )

    raw_coordinates = (
        geometry.get(
            "coordinates",
            []
        )
        if isinstance(
            geometry,
            dict
        )
        else []
    )

    # OSRM returns [longitude, latitude].
    # Frontend receives [latitude, longitude].

    route_coordinates = []

    for coordinate in raw_coordinates:

        if (
            isinstance(coordinate, list)
            and
            len(coordinate) >= 2
        ):

            route_coordinates.append([
                coordinate[1],
                coordinate[0]
            ])

    return {

        "distance_meters":
            distance_meters,

        "duration_seconds":
            duration_seconds,

        "route_coordinates":
            route_coordinates,

        "provider":
            "OSRM"
    }


# ============================================================
# ROUTE NORMALIZATION
# ============================================================

def _duration_seconds(
    duration_value
):

    if duration_value is None:
        return None

    if isinstance(
        duration_value,
        (int, float)
    ):
        return float(
            duration_value
        )

    value = str(
        duration_value
    ).strip()

    if value.endswith("s"):

        try:
            return float(
                value[:-1]
            )

        except ValueError:
            return None

    try:
        return float(
            value
        )

    except ValueError:
        return None


def _format_duration(
    seconds
):

    if seconds is None:
        return None

    seconds = int(
        round(seconds)
    )

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    if hours > 0:

        return (
            f"{hours} hr "
            f"{minutes} min"
        )

    if minutes <= 0:
        return "<1 min"

    return f"{minutes} min"


# ============================================================
# GOOGLE POLYLINE DECODER
# ============================================================

def _decode_polyline(
    encoded
):

    if not encoded:
        return []

    coordinates = []

    index = 0

    latitude = 0

    longitude = 0

    while index < len(encoded):

        result = 0
        shift = 0

        while True:

            if index >= len(encoded):
                return coordinates

            byte = (
                ord(encoded[index])
                - 63
            )

            index += 1

            result |= (
                (byte & 0x1F)
                << shift
            )

            shift += 5

            if byte < 0x20:
                break

        latitude_change = (
            -(result >> 1)
            if result & 1
            else result >> 1
        )

        latitude += latitude_change

        result = 0
        shift = 0

        while True:

            if index >= len(encoded):
                return coordinates

            byte = (
                ord(encoded[index])
                - 63
            )

            index += 1

            result |= (
                (byte & 0x1F)
                << shift
            )

            shift += 5

            if byte < 0x20:
                break

        longitude_change = (
            -(result >> 1)
            if result & 1
            else result >> 1
        )

        longitude += longitude_change

        coordinates.append([
            latitude / 100000.0,
            longitude / 100000.0
        ])

    return coordinates


# ============================================================
# DETERMINISTIC ROUTE SCORING
# ============================================================

def _deterministic_recommendation(
    candidates
):

    if not candidates:
        return None

    usable = [

        candidate

        for candidate in candidates

        if (
            candidate.get(
                "route_distance_km"
            ) is not None
            and
            candidate.get(
                "duration_seconds"
            ) is not None
        )
    ]

    if not usable:
        return None

    max_distance = max(
        candidate[
            "route_distance_km"
        ]
        for candidate in usable
    ) or 1

    max_duration = max(
        candidate[
            "duration_seconds"
        ]
        for candidate in usable
    ) or 1

    for candidate in usable:

        distance_score = (
            1
            -
            (
                candidate[
                    "route_distance_km"
                ]
                /
                max_distance
            )
        )

        duration_score = (
            1
            -
            (
                candidate[
                    "duration_seconds"
                ]
                /
                max_duration
            )
        )

        score = (
            distance_score * 0.55
            +
            duration_score * 0.45
        )

        candidate[
            "safety_score"
        ] = round(
            70 + score * 29,
            1
        )

    usable.sort(

        key=lambda item: (

            -item[
                "safety_score"
            ],

            item[
                "route_distance_km"
            ],

            item[
                "duration_seconds"
            ]
        )
    )

    return usable[0]


# ============================================================
# GEMINI DIRECT API
# ============================================================

def _extract_json_from_text(
    text
):

    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):

        cleaned = cleaned.replace(
            "```json",
            "",
            1
        )

        cleaned = cleaned.replace(
            "```",
            "",
            1
        ).strip()

    first = cleaned.find("{")

    last = cleaned.rfind("}")

    if (
        first == -1
        or
        last == -1
        or
        last <= first
    ):
        return None

    try:

        return json.loads(
            cleaned[
                first:last + 1
            ]
        )

    except json.JSONDecodeError:

        return None


def _call_gemini(
    candidates
):

    if not GEMINI_API_KEY:
        return None

    candidate_payload = []

    for candidate in candidates:

        candidate_payload.append({

            "rank":
                candidate.get(
                    "rank"
                ),

            "name":
                candidate.get(
                    "name"
                ),

            "address":
                candidate.get(
                    "address"
                ),

            "road_distance_km":
                candidate.get(
                    "route_distance_km"
                ),

            "eta_minutes":
                candidate.get(
                    "duration_minutes"
                ),

            "straight_line_distance_km":
                candidate.get(
                    "distance_km"
                )

        })

    prompt = """
You are FloodGuard AI, an emergency evacuation
recommendation assistant.

Select the safest PRACTICAL shelter from the
candidate shelters.

Only use facts supplied in the candidate data.

Do NOT invent:
- flood risk
- capacity
- availability
- road closures
- emergency conditions

Use:
- actual road distance
- driving ETA
- straight-line distance

Prefer shorter road distance and shorter ETA.

Return ONLY valid JSON in exactly this format:

{
  "recommended_rank": 1,
  "priority": "HIGH",
  "confidence": 90,
  "summary": "Short explanation.",
  "reasons": [
    "Reason one.",
    "Reason two.",
    "Reason three."
  ]
}

Candidate shelters:

""" + json.dumps(
        candidate_payload,
        ensure_ascii=False
    )

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
        f"?key={urllib.parse.quote(GEMINI_API_KEY)}"
    )

    payload = {

        "contents": [

            {

                "parts": [

                    {
                        "text": prompt
                    }

                ]

            }

        ],

        "generationConfig": {

            "temperature": 0.1,

            "responseMimeType":
                "application/json"

        }

    }

    request = urllib.request.Request(

        url,

        data=json.dumps(
            payload
        ).encode("utf-8"),

        method="POST",

        headers={
            "Content-Type":
                "application/json"
        }

    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=45
        ) as response:

            raw = response.read().decode(
                "utf-8",
                errors="replace"
            )

            data = json.loads(
                raw
            )

    except urllib.error.HTTPError as e:

        error_body = e.read().decode(
            "utf-8",
            errors="replace"
        )

        print(
            "Gemini API error:",
            e.code,
            error_body[:1000]
        )

        return None

    except Exception as e:

        print(
            "Gemini request failed:",
            str(e)
        )

        return None

    try:

        generated_candidates = data.get(
            "candidates",
            []
        )

        if not generated_candidates:
            return None

        content = (
            generated_candidates[0]
            .get(
                "content",
                {}
            )
        )

        parts = content.get(
            "parts",
            []
        )

        if not parts:
            return None

        text = parts[0].get(
            "text",
            ""
        )

        return _extract_json_from_text(
            text
        )

    except Exception:

        return None


# ============================================================
# BUILD RECOMMENDATION
# ============================================================

def _build_recommendation(
    candidates
):

    deterministic = (
        _deterministic_recommendation(
            candidates
        )
    )

    if not deterministic:

        return {

            "priority":
                "UNKNOWN",

            "confidence":
                0,

            "summary":
                "No reliable route recommendation available.",

            "reasons":
                [],

            "engine":
                "FloodGuard route scoring",

            "recommended_rank":
                None,

            "ai_enabled":
                False
        }

    ai_result = _call_gemini(
        candidates
    )

    if ai_result:

        try:

            recommended_rank = int(
                ai_result.get(
                    "recommended_rank"
                )
            )

        except (
            TypeError,
            ValueError
        ):

            recommended_rank = (
                deterministic.get(
                    "rank"
                )
            )

        selected = next(

            (
                candidate

                for candidate
                in candidates

                if candidate.get(
                    "rank"
                )
                ==
                recommended_rank
            ),

            None
        )

        if selected:

            priority = str(
                ai_result.get(
                    "priority",
                    "HIGH"
                )
            ).upper()

            if priority not in {
                "HIGH",
                "MEDIUM",
                "LOW"
            }:

                priority = "HIGH"

            try:

                confidence = float(
                    ai_result.get(
                        "confidence",
                        selected.get(
                            "safety_score",
                            0
                        )
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                confidence = selected.get(
                    "safety_score",
                    0
                )

            confidence = max(
                0,
                min(
                    100,
                    confidence
                )
            )

            reasons = ai_result.get(
                "reasons",
                []
            )

            if not isinstance(
                reasons,
                list
            ):

                reasons = []

            return {

                "priority":
                    priority,

                "confidence":
                    round(
                        confidence,
                        1
                    ),

                "summary":
                    ai_result.get(
                        "summary",
                        "Recommended using route distance and ETA."
                    ),

                "reasons":
                    reasons[:5],

                "engine":
                    "FloodGuard + Gemini",

                "recommended_rank":
                    selected.get(
                        "rank"
                    ),

                "ai_enabled":
                    True
            }

    return {

        "priority":
            "HIGH",

        "confidence":
            deterministic.get(
                "safety_score",
                0
            ),

        "summary":
            (
                "Recommended using actual road distance "
                "and driving ETA."
            ),

        "reasons": [

            "Shortest practical road route among candidates.",

            "Driving ETA is included in the recommendation.",

            "No unsupported shelter capacity or flood-risk "
            "information was assumed."
        ],

        "engine":
            "FloodGuard route scoring",

        "recommended_rank":
            deterministic.get(
                "rank"
            ),

        "ai_enabled":
            False
    }


# ============================================================
# SAFE ROUTE
# ============================================================

@app.get("/safe-route")
async def safe_route(
    latitude: float,
    longitude: float
):

    if not -90 <= latitude <= 90:

        raise HTTPException(
            status_code=400,
            detail="Invalid latitude."
        )

    if not -180 <= longitude <= 180:

        raise HTTPException(
            status_code=400,
            detail="Invalid longitude."
        )

    # --------------------------------------------------------
    # 1. Find real nearby shelters
    # --------------------------------------------------------

    shelters = _get_shelters_from_floodguard_database(
        latitude,
        longitude
    )

    if not shelters:

        return {

            "success":
                False,

            "message":
                "No emergency shelters were found near this location.",

            "origin": {

                "latitude":
                    latitude,

                "longitude":
                    longitude

            },

            "recommended_shelter":
                None,

            "route":
                None,

            "shelters":
                [],

            "recommendation":
                None
        }

    # --------------------------------------------------------
    # 2. Take nearest candidates
    # --------------------------------------------------------

    valid_shelters = [

        shelter

        for shelter in shelters

        if (
            shelter.get(
                "latitude"
            ) is not None
            and
            shelter.get(
                "longitude"
            ) is not None
        )
    ]

    if not valid_shelters:

        return {

            "success":
                False,

            "message":
                "Shelter coordinates are unavailable.",

            "origin": {

                "latitude":
                    latitude,

                "longitude":
                    longitude

            },

            "recommended_shelter":
                None,

            "route":
                None,

            "shelters":
                shelters,

            "recommendation":
                None
        }

    # Already sorted by straight-line distance.
    candidates = valid_shelters[:5]

    routed_candidates = []

    # --------------------------------------------------------
    # 3. Calculate route for each candidate
    # --------------------------------------------------------

    for index, shelter in enumerate(
        candidates
    ):

        destination_latitude = float(
            shelter[
                "latitude"
            ]
        )

        destination_longitude = float(
            shelter[
                "longitude"
            ]
        )

        routed = dict(
            shelter
        )

        routed[
            "rank"
        ] = index + 1

        google_result = (
            _google_routes_request(

                latitude,

                longitude,

                destination_latitude,

                destination_longitude

            )
        )

        # ----------------------------------------------------
        # GOOGLE ROUTES
        # ----------------------------------------------------

        if google_result:

            google_route = (
                google_result[
                    "route"
                ]
            )

            distance_meters = (
                google_route.get(
                    "distanceMeters"
                )
            )

            duration_seconds = (
                _duration_seconds(
                    google_route.get(
                        "duration"
                    )
                )
            )

            encoded_polyline = (

                google_route
                .get(
                    "polyline",
                    {}
                )
                .get(
                    "encodedPolyline"
                )
            )

            route_coordinates = (
                _decode_polyline(
                    encoded_polyline
                )
            )

            road_distance_km = (

                float(
                    distance_meters
                ) / 1000.0

                if distance_meters is not None

                else None
            )

            routed[
                "route_distance_km"
            ] = (

                round(
                    road_distance_km,
                    2
                )

                if road_distance_km is not None

                else None
            )

            routed[
                "duration_seconds"
            ] = duration_seconds

            routed[
                "duration_minutes"
            ] = (

                round(
                    duration_seconds / 60,
                    1
                )

                if duration_seconds is not None

                else None
            )

            routed[
                "duration_text"
            ] = _format_duration(
                duration_seconds
            )

            routed[
                "route_polyline"
            ] = encoded_polyline

            routed[
                "route_coordinates"
            ] = route_coordinates

            routed[
                "route_status"
            ] = "AVAILABLE"

            routed[
                "route_provider"
            ] = "Google Routes API"

        else:

            # ------------------------------------------------
            # GOOGLE FAILED → OSRM FALLBACK
            # ------------------------------------------------

            osrm_result = (
                _osrm_route_request(

                    latitude,

                    longitude,

                    destination_latitude,

                    destination_longitude

                )
            )

            if osrm_result:

                distance_meters = (
                    osrm_result.get(
                        "distance_meters"
                    )
                )

                duration_seconds = (
                    osrm_result.get(
                        "duration_seconds"
                    )
                )

                routed[
                    "route_distance_km"
                ] = (

                    round(
                        float(
                            distance_meters
                        ) / 1000.0,
                        2
                    )

                    if distance_meters is not None

                    else None
                )

                routed[
                    "duration_seconds"
                ] = duration_seconds

                routed[
                    "duration_minutes"
                ] = (

                    round(
                        float(
                            duration_seconds
                        ) / 60,
                        1
                    )

                    if duration_seconds is not None

                    else None
                )

                routed[
                    "duration_text"
                ] = _format_duration(
                    duration_seconds
                )

                routed[
                    "route_polyline"
                ] = None

                routed[
                    "route_coordinates"
                ] = osrm_result.get(
                    "route_coordinates",
                    []
                )

                routed[
                    "route_status"
                ] = "AVAILABLE"

                routed[
                    "route_provider"
                ] = "OSRM"

            else:

                routed[
                    "route_distance_km"
                ] = None

                routed[
                    "duration_seconds"
                ] = None

                routed[
                    "duration_minutes"
                ] = None

                routed[
                    "duration_text"
                ] = None

                routed[
                    "route_polyline"
                ] = None

                routed[
                    "route_coordinates"
                ] = []

                routed[
                    "route_status"
                ] = "UNAVAILABLE"

        routed_candidates.append(
            routed
        )

    # --------------------------------------------------------
    # 4. Only candidates with routes
    # --------------------------------------------------------

    usable_candidates = [

        candidate

        for candidate
        in routed_candidates

        if (
            candidate.get(
                "route_distance_km"
            ) is not None

            and

            candidate.get(
                "duration_seconds"
            ) is not None
        )
    ]

    if not usable_candidates:

        return {

            "success":
                False,

            "message": (
                "Shelters were found, but no driving route "
                "could be calculated."
            ),

            "origin": {

                "latitude":
                    latitude,

                "longitude":
                    longitude

            },

            "recommended_shelter":
                None,

            "route":
                None,

            "shelters":
                routed_candidates,

            "recommendation":
                None
        }

    # --------------------------------------------------------
    # 5. Recommendation
    # --------------------------------------------------------

    recommendation = (
        _build_recommendation(
            usable_candidates
        )
    )

    recommended_rank = (
        recommendation.get(
            "recommended_rank"
        )
    )

    recommended = next(

        (

            candidate

            for candidate
            in usable_candidates

            if candidate.get(
                "rank"
            )
            ==
            recommended_rank

        ),

        None
    )

    if recommended is None:

        recommended = min(

            usable_candidates,

            key=lambda item: (

                item.get(
                    "route_distance_km",
                    float("inf")
                ),

                item.get(
                    "duration_seconds",
                    float("inf")
                )

            )
        )

    # --------------------------------------------------------
    # 6. Rank all shelters
    # --------------------------------------------------------

    ranked_candidates = [

        dict(candidate)

        for candidate
        in usable_candidates
    ]

    _deterministic_recommendation(
        ranked_candidates
    )

    ranked_candidates.sort(

        key=lambda item: (

            -float(
                item.get(
                    "safety_score",
                    0
                )
            ),

            item.get(
                "route_distance_km",
                float("inf")
            )
        )
    )

    final_shelters = []

    for position, shelter in enumerate(
        ranked_candidates
    ):

        item = dict(
            shelter
        )

        item[
            "recommendation_rank"
        ] = position + 1

        if (
            shelter.get(
                "id"
            )
            ==
            recommended.get(
                "id"
            )
        ):

            item[
                "recommendation"
            ] = "RECOMMENDED"

        elif position == 1:

            item[
                "recommendation"
            ] = "ALTERNATIVE"

        else:

            item[
                "recommendation"
            ] = "BACKUP"

        final_shelters.append(
            item
        )

    # --------------------------------------------------------
    # 7. Google Maps navigation URL
    # --------------------------------------------------------

    origin = (
        f"{latitude},{longitude}"
    )

    destination = (
        f"{recommended['latitude']},"
        f"{recommended['longitude']}"
    )

    maps_url = (

        "https://www.google.com/maps/dir/?api=1"

        f"&origin="
        f"{urllib.parse.quote(origin)}"

        f"&destination="
        f"{urllib.parse.quote(destination)}"

        "&travelmode=driving"
    )

    # --------------------------------------------------------
    # 8. Route object
    # --------------------------------------------------------

    route = {

        "type":
            "driving",

        "status":
            "AVAILABLE",

        "distance_km":
            recommended.get(
                "route_distance_km"
            ),

        "duration_seconds":
            recommended.get(
                "duration_seconds"
            ),

        "duration_minutes":
            recommended.get(
                "duration_minutes"
            ),

        "duration_text":
            recommended.get(
                "duration_text"
            ),

        "polyline":
            recommended.get(
                "route_polyline"
            ),

        "coordinates":
            recommended.get(
                "route_coordinates",
                []
            ),

        "navigation_url":
            maps_url,

        "provider":
            recommended.get(
                "route_provider"
            )
    }

    # --------------------------------------------------------
    # 9. Final response
    # --------------------------------------------------------

    return {

        "success":
            True,

        "origin": {

            "latitude":
                latitude,

            "longitude":
                longitude
        },

        "recommended_shelter": {

            "id":
                recommended.get(
                    "id"
                ),

            "name":
                recommended.get(
                    "name"
                ),

            "address":
                recommended.get(
                    "address"
                ),

            "latitude":
                recommended.get(
                    "latitude"
                ),

            "longitude":
                recommended.get(
                    "longitude"
                ),

            "straight_line_distance_km":
                recommended.get(
                    "distance_km"
                ),

            "road_distance_km":
                recommended.get(
                    "route_distance_km"
                ),

            "eta_minutes":
                recommended.get(
                    "duration_minutes"
                ),

            "eta_text":
                recommended.get(
                    "duration_text"
                ),

            "safety_score":
                recommended.get(
                    "safety_score"
                ),

            "maps_url":
                maps_url
        },

        "route":
            route,

        "shelters":
            final_shelters,

        "recommendation":
            recommendation,

        "data_sources": {

            "shelters":
                "FloodGuard Supabase shelters table",

            "route":
                recommended.get(
                    "route_provider"
                ),

            "recommendation":
                recommendation.get(
                    "engine"
                )
        }
    }
# ============================================================
# MODULE 5 / 7 / 8 — OPERATIONAL OPTIMIZATION
# ============================================================

@app.post("/api/module5/run")
async def trigger_module5():
    try:
        assignments = run_module5()
        return {"success": True, "assignments": assignments, "count": len(assignments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Module 5 failed: {str(e)}")


@app.post("/api/module7/run")
async def trigger_module7():
    try:
        assignments = run_module7()
        return {"success": True, "assignments": assignments, "count": len(assignments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Module 7 failed: {str(e)}")


@app.post("/api/module8/run")
async def trigger_module8():
    try:
        assessments = run_module8()
        return {"success": True, "assessments": assessments, "count": len(assessments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Module 8 failed: {str(e)}")

# ============================================================
# MODULE 5 / 7 / 8 — LIVE DATA READS (for frontend dashboards)
# ============================================================

@app.get("/api/rescue-resources")
async def get_rescue_resources():
    try:
        incidents = operations_supabase.table("rescue_incidents").select("*").execute().data
        resources = operations_supabase.table("rescue_resources").select("*").execute().data
        assignments = operations_supabase.table("rescue_assignments").select("*").execute().data
        return {"incidents": incidents, "resources": resources, "assignments": assignments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load rescue data: {str(e)}")


@app.get("/api/shelters-live")
async def get_shelters_live():
    try:
        shelters = operations_supabase.table("shelters").select("*").execute().data
        return {"shelters": shelters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load shelter data: {str(e)}")


@app.get("/api/relief-status")
async def get_relief_status():
    try:
        assessments = (
            operations_supabase.table("relief_status_view")
            .select("*")
            .order("shelter_id")
            .execute()
            .data
        )
        budgets = operations_supabase.table("budgets").select("*").execute().data
        expenses = operations_supabase.table("expenses").select("*").execute().data
        total_allocated = sum(float(b["total_allocated"]) for b in budgets)
        total_spent = sum(float(e["amount"]) for e in expenses)
        return {
            "assessments": assessments,
            "allocated": total_allocated,
            "spent": total_spent,
            "remaining": total_allocated - total_spent,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load relief data: {str(e)}")


def _thingspeak_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


@app.get("/api/thingspeak/sensor-data")
async def get_thingspeak_sensor_data():
    readings = _read_thingspeak_readings()
    public_readings = [
        {key: value for key, value in reading.items() if key != "entry_id"}
        for reading in readings
    ]
    return {
        "readings": public_readings,
        "latest": public_readings[0] if public_readings else None,
    }


def _read_thingspeak_readings():
    channel_id = os.getenv("THINGSPEAK_CHANNEL_ID")
    read_api_key = os.getenv("THINGSPEAK_READ_API_KEY")

    if not channel_id or not read_api_key:
        raise HTTPException(
            status_code=500,
            detail="ThingSpeak channel configuration is missing",
        )

    query = urllib.parse.urlencode({
        "api_key": read_api_key,
        "results": 20,
    })
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?{query}"

    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"ThingSpeak request failed: {error}",
        )

    readings = []
    for feed in payload.get("feeds", []):
        reading = {
            "entry_id": feed.get("entry_id"),
            "created_at": feed.get("created_at"),
            "water_distance_cm": _thingspeak_number(feed.get("field1")),
            "water_rate_cm_per_min": _thingspeak_number(feed.get("field2")),
            "soil_moisture_pct": _thingspeak_number(feed.get("field3")),
            "temperature_c": _thingspeak_number(feed.get("field4")),
            "humidity_pct": _thingspeak_number(feed.get("field5")),
            "vibration_index": _thingspeak_number(feed.get("field6")),
            "latitude": _thingspeak_number(feed.get("field7")),
            "longitude": _thingspeak_number(feed.get("field8")),
        }
        if reading["created_at"]:
            readings.append(reading)

    readings.reverse()
    return readings


def _parse_thingspeak_timestamp(value):
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _configured_flood_float(name):
    value = os.getenv(name)
    if value is None or not value.strip():
        raise HTTPException(
            status_code=500,
            detail=f"Required configuration {name} is missing.",
        )

    try:
        parsed = float(value)
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Configuration {name} must be numeric.",
        ) from error

    if not math.isfinite(parsed):
        raise HTTPException(
            status_code=500,
            detail=f"Configuration {name} must be a finite number.",
        )
    return parsed


def _latest_valid_thingspeak_reading(readings):
    required_fields = (
        "water_distance_cm",
        "water_rate_cm_per_min",
        "soil_moisture_pct",
    )
    timestamped_readings = [
        (reading, _parse_thingspeak_timestamp(reading.get("created_at")))
        for reading in readings
    ]
    timestamped_readings = [
        (reading, timestamp)
        for reading, timestamp in timestamped_readings
        if timestamp is not None
    ]

    valid_readings = [
        (reading, timestamp)
        for reading, timestamp in timestamped_readings
        if all(
            isinstance(reading.get(field), (int, float))
            and math.isfinite(reading[field])
            for field in required_fields
        )
    ]
    if valid_readings:
        return max(valid_readings, key=lambda item: item[1])[0]

    if timestamped_readings:
        newest = max(timestamped_readings, key=lambda item: item[1])[0]
        missing_fields = [
            field
            for field in required_fields
            if not isinstance(newest.get(field), (int, float))
            or not math.isfinite(newest[field])
        ]
        raise HTTPException(
            status_code=422,
            detail=(
                "Newest ThingSpeak entry has missing or non-numeric required "
                f"sensor field(s): {', '.join(missing_fields)}."
            ),
        )

    raise HTTPException(
        status_code=502,
        detail="ThingSpeak returned no readings with a valid created_at timestamp.",
    )


def _live_flood_model_inputs(reading, water_sensor_height_cm, rainfall_mm_per_hr, elevation_m):
    distance_cm = reading["water_distance_cm"]
    distance_rate_cm_per_min = reading["water_rate_cm_per_min"]
    soil_moisture_pct = reading["soil_moisture_pct"]

    water_level_cm = water_sensor_height_cm - distance_cm
    # Arduino field2 is distance-change rate: shrinking distance means rising water.
    water_level_rate_cm_per_min = -distance_rate_cm_per_min

    values = {
        "water_level_cm": water_level_cm,
        "water_level_rate_cm_per_min": water_level_rate_cm_per_min,
        "rainfall_mm_per_hr": rainfall_mm_per_hr,
        "soil_moisture_pct": soil_moisture_pct,
        "elevation_m": elevation_m,
    }
    invalid_values = [
        name for name, value in values.items() if not math.isfinite(value)
    ]
    if invalid_values:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid calculated Model 1 input(s): {', '.join(invalid_values)}.",
        )
    if elevation_m + 1.0 == 0:
        raise HTTPException(
            status_code=422,
            detail="FLOOD_ELEVATION_M produces an invalid Model 1 calculation.",
        )
    if soil_moisture_pct + 1.0 == 0:
        raise HTTPException(
            status_code=422,
            detail="ThingSpeak soil_moisture_pct produces an invalid Model 1 calculation.",
        )
    return values


@app.get("/flood-risk/live")
async def live_flood_risk():
    reading = _latest_valid_thingspeak_reading(_read_thingspeak_readings())
    water_sensor_height_cm = _configured_flood_float("WATER_SENSOR_HEIGHT_CM")
    rainfall_mm_per_hr = _configured_flood_float("FLOOD_RAINFALL_MM_PER_HR")
    elevation_m = _configured_flood_float("FLOOD_ELEVATION_M")

    model_inputs = _live_flood_model_inputs(
        reading,
        water_sensor_height_cm,
        rainfall_mm_per_hr,
        elevation_m,
    )
    try:
        prediction = predict_flood_risk(**model_inputs)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Live flood-risk inference failed: {error}",
        ) from error

    return {
        "success": True,
        "flood_prediction": prediction,
        "model_inputs": model_inputs,
        "input_sources": {
            "water_level": "ThingSpeak field1 + WATER_SENSOR_HEIGHT_CM",
            "water_level_rate": "negative of ThingSpeak field2",
            "soil_moisture": "ThingSpeak field3",
            "rainfall": "environment_configured",
            "elevation": "environment_configured",
        },
        "sensor_data": {
            "temperature_c": reading.get("temperature_c"),
            "humidity_pct": reading.get("humidity_pct"),
            "vibration_index": reading.get("vibration_index"),
            "latitude": reading.get("latitude"),
            "longitude": reading.get("longitude"),
        },
        "thingspeak": {
            "entry_id": reading.get("entry_id"),
            "created_at": reading["created_at"],
        },
    }