import os
import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from fastapi import HTTPException

from backend.main import (
    app,
    _latest_valid_thingspeak_reading,
    _live_flood_model_inputs,
    _read_thingspeak_readings,
)
from backend.services.flood_risk import predict_flood_risk


class FloodRiskLiveTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.reading = {
            "entry_id": 123,
            "created_at": "2026-09-27T12:00:00Z",
            "water_distance_cm": 30.0,
            "water_rate_cm_per_min": -0.5,
            "soil_moisture_pct": 65.0,
            "temperature_c": 29.4,
            "humidity_pct": 82.0,
            "vibration_index": 0.13,
            "latitude": None,
            "longitude": None,
        }

    def test_water_level_rate_and_soil_conversions(self):
        values = _live_flood_model_inputs(self.reading, 100.0, 18.7, 214.6)
        self.assertEqual(values["water_level_cm"], 70.0)
        self.assertEqual(values["water_level_rate_cm_per_min"], 0.5)
        self.assertEqual(values["soil_moisture_pct"], 65.0)

    def test_positive_distance_rate_means_falling_water(self):
        reading = {**self.reading, "water_rate_cm_per_min": 0.5}
        values = _live_flood_model_inputs(reading, 100.0, 18.7, 214.6)
        self.assertEqual(values["water_level_rate_cm_per_min"], -0.5)

    def test_invalid_elevation_calculation_rejected(self):
        with self.assertRaises(HTTPException) as raised:
            _live_flood_model_inputs(self.reading, 100.0, 18.7, -1.0)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("FLOOD_ELEVATION_M", raised.exception.detail)

        reading = {**self.reading, "soil_moisture_pct": -1.0}
        with self.assertRaises(HTTPException) as raised:
            _live_flood_model_inputs(reading, 100.0, 18.7, 214.6)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("soil_moisture_pct", raised.exception.detail)

    def test_existing_thingspeak_parser_maps_fields_and_entry_id(self):
        payload = {
            "feeds": [{
                "entry_id": 123,
                "created_at": "2026-09-27T12:00:00Z",
                "field1": "30",
                "field2": "-0.5",
                "field3": "65",
                "field4": "29.4",
                "field5": "82",
                "field6": "0.13",
                "field7": "12.5",
                "field8": "77.2",
            }]
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        with (
            patch.dict(os.environ, {
                "THINGSPEAK_CHANNEL_ID": "test-channel",
                "THINGSPEAK_READ_API_KEY": "test-read-key",
            }),
            patch("backend.main.urllib.request.urlopen", return_value=response),
        ):
            readings = _read_thingspeak_readings()

        self.assertEqual(len(readings), 1)
        self.assertEqual(readings[0]["entry_id"], 123)
        self.assertEqual(readings[0]["water_distance_cm"], 30.0)
        self.assertEqual(readings[0]["water_rate_cm_per_min"], -0.5)
        self.assertEqual(readings[0]["soil_moisture_pct"], 65.0)
        self.assertEqual(readings[0]["temperature_c"], 29.4)
        self.assertEqual(readings[0]["humidity_pct"], 82.0)
        self.assertEqual(readings[0]["vibration_index"], 0.13)
        self.assertEqual(readings[0]["latitude"], 12.5)
        self.assertEqual(readings[0]["longitude"], 77.2)

    def test_latest_valid_reading_selected_by_timestamp(self):
        older = {**self.reading, "created_at": "2026-09-27T11:00:00Z"}
        newer = {**self.reading, "entry_id": 124, "created_at": "2026-09-27T12:00:00Z"}
        selected = _latest_valid_thingspeak_reading([newer, older])
        self.assertEqual(selected["entry_id"], 124)

        invalid_newest = {
            **newer,
            "entry_id": 125,
            "created_at": "2026-09-27T13:00:00Z",
            "water_distance_cm": None,
        }
        selected = _latest_valid_thingspeak_reading([invalid_newest, newer, older])
        self.assertEqual(selected["entry_id"], 124)

    def test_missing_or_non_numeric_required_field_rejected(self):
        for field, invalid_value in (
            ("water_distance_cm", None),
            ("water_rate_cm_per_min", "not-a-number"),
            ("soil_moisture_pct", None),
        ):
            with self.subTest(field=field):
                invalid = {**self.reading, field: invalid_value}
                with self.assertRaises(HTTPException) as raised:
                    _latest_valid_thingspeak_reading([invalid])
                self.assertEqual(raised.exception.status_code, 422)
                self.assertIn(field, raised.exception.detail)

    def test_missing_configuration_is_clear(self):
        configured_values = {
            "WATER_SENSOR_HEIGHT_CM": "100",
            "FLOOD_RAINFALL_MM_PER_HR": "18.7",
            "FLOOD_ELEVATION_M": "214.6",
        }
        for name in (
            "WATER_SENSOR_HEIGHT_CM",
            "FLOOD_RAINFALL_MM_PER_HR",
            "FLOOD_ELEVATION_M",
        ):
            with self.subTest(name=name):
                environment = {**configured_values, name: ""}
                with (
                    patch.dict(os.environ, environment),
                    patch("backend.main._read_thingspeak_readings", return_value=[self.reading]),
                ):
                    response = self.client.get("/flood-risk/live")
                self.assertEqual(response.status_code, 500)
                self.assertIn(name, response.json()["detail"])

    def test_invalid_configuration_rejected(self):
        with patch.dict(os.environ, {"WATER_SENSOR_HEIGHT_CM": "nan"}):
            from backend.main import _configured_flood_float
            with self.assertRaises(HTTPException) as raised:
                _configured_flood_float("WATER_SENSOR_HEIGHT_CM")
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIn("finite", raised.exception.detail)

    def test_existing_model_prediction_succeeds(self):
        prediction = predict_flood_risk(
            water_level_cm=82.4,
            water_level_rate_cm_per_min=0.43,
            rainfall_mm_per_hr=18.7,
            soil_moisture_pct=67.3,
            elevation_m=214.6,
        )
        self.assertIn("risk_class", prediction)
        self.assertIn("risk_label", prediction)
        self.assertGreaterEqual(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 1.0)

    def test_live_endpoint_response_with_mocked_thingspeak(self):
        with (
            patch("backend.main._read_thingspeak_readings", return_value=[self.reading]),
            patch.dict(os.environ, {
                "WATER_SENSOR_HEIGHT_CM": "100",
                "FLOOD_RAINFALL_MM_PER_HR": "18.7",
                "FLOOD_ELEVATION_M": "214.6",
            }),
        ):
            response = self.client.get("/flood-risk/live")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["model_inputs"]["water_level_cm"], 70.0)
        self.assertEqual(payload["model_inputs"]["water_level_rate_cm_per_min"], 0.5)
        self.assertEqual(payload["model_inputs"]["soil_moisture_pct"], 65.0)
        self.assertEqual(payload["input_sources"]["rainfall"], "environment_configured")
        self.assertEqual(payload["input_sources"]["elevation"], "environment_configured")
        self.assertEqual(payload["sensor_data"]["temperature_c"], 29.4)
        self.assertEqual(payload["sensor_data"]["humidity_pct"], 82.0)
        self.assertEqual(payload["sensor_data"]["vibration_index"], 0.13)
        self.assertEqual(payload["thingspeak"]["entry_id"], 123)
        self.assertEqual(payload["thingspeak"]["created_at"], self.reading["created_at"])
        self.assertIn("risk_label", payload["flood_prediction"])
        self.assertIn("confidence", payload["flood_prediction"])

    def test_legacy_thingspeak_response_shape_unchanged(self):
        with patch("backend.main._read_thingspeak_readings", return_value=[self.reading]):
            response = self.client.get("/api/thingspeak/sensor-data")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload), {"readings", "latest"})
        self.assertNotIn("entry_id", payload["latest"])


if __name__ == "__main__":
    unittest.main()
