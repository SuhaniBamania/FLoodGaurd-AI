"use client"

import { useEffect, useMemo, useState } from "react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import {
  Activity,
  CheckCircle2,
  Droplet,
  Gauge,
  Leaf,
  Megaphone,
  Radio,
  Thermometer,
  Wind,
  Zap,
} from "lucide-react"
import { apiGet } from "@/lib/api"

/*
 * IMPORTANT:
 * This Dashboard component intentionally DOES NOT render a sidebar or top header.
 * Your application layout should render those globally.
 * This prevents the duplicate header/sidebar problem visible in the screenshot.
 *
 * Keep your existing API/Supabase/model hooks in the marked data section below.
 * The UI structure is independent of those data sources.
 */

const SENSOR_DATA = {
  waterLevel: 2.35,
  rainfall: 24.7,
  soilMoisture: 78,
  temperature: 28.5,
  movement: 0.12,
  waterFlow: 3.4,
}

type FloodRiskLiveResponse = {
  success: boolean
  flood_prediction: {
    risk_class: number
    risk_label: string
    confidence: number
  }
  model_inputs: {
    water_level_cm: number
    water_level_rate_cm_per_min: number
    rainfall_mm_per_hr: number
    soil_moisture_pct: number
    elevation_m: number
  }
  input_sources: {
    water_level: string
    water_level_rate: string
    soil_moisture: string
    rainfall: string
    elevation: string
  }
  sensor_data: {
    temperature_c: number | null
    humidity_pct: number | null
    vibration_index: number | null
    latitude: number | null
    longitude: number | null
  }
  thingspeak: {
    entry_id: number | null
    created_at: string | null
  }
}

type SensorPoint = {
  time: string
  waterLevel?: number
  soilMoisture?: number
  temperature?: number
  movement?: number
  waterFlow?: number
}

type SensorKey =
  | "waterLevel"
  | "waterFlow"
  | "soilMoisture"
  | "temperature"
  | "movement"

const SENSOR_META: Record<SensorKey, { title: string; unit: string }> = {
  waterLevel: { title: "Water Level", unit: "cm" },
  waterFlow: { title: "Water Flow", unit: "cm/min" },
  soilMoisture: { title: "Soil Moisture", unit: "%" },
  temperature: { title: "Temperature", unit: "°C" },
  movement: { title: "Ground Movement", unit: "vibration index" },
}

export default function Dashboard() {
  const [currentTime, setCurrentTime] = useState<Date | null>(null)
  const [selectedSensor, setSelectedSensor] = useState<SensorKey>("waterLevel")
  const [sensorHistory, setSensorHistory] = useState<SensorPoint[]>([])
  const [floodRiskData, setFloodRiskData] =
    useState<FloodRiskLiveResponse | null>(null)
  const [floodRiskLoading, setFloodRiskLoading] = useState(true)
  const [floodRiskError, setFloodRiskError] = useState<string | null>(null)

  useEffect(() => {
    setCurrentTime(new Date())
    const timer = window.setInterval(() => setCurrentTime(new Date()), 1000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    let cancelled = false

    async function loadSensorData() {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_FLOODGAURD_API_URL || "http://localhost:8000"}/api/thingspeak/sensor-data`,
          { cache: "no-store" },
        )
        if (!response.ok) return

        const payload = await response.json()
        const points: SensorPoint[] = (payload.readings || []).map(
          (reading: Record<string, unknown>) => ({
            time: new Date(String(reading.created_at)).toLocaleTimeString("en-IN", {
              hour: "2-digit",
              minute: "2-digit",
            }),
            ...(typeof reading.water_distance_cm === "number" && {
              waterLevel: reading.water_distance_cm,
            }),
            ...(typeof reading.soil_moisture_pct === "number" && {
              soilMoisture: reading.soil_moisture_pct,
            }),
            ...(typeof reading.temperature_c === "number" && {
              temperature: reading.temperature_c,
            }),
            ...(typeof reading.vibration_index === "number" && {
              movement: reading.vibration_index,
            }),
            ...(typeof reading.water_rate_cm_per_min === "number" && {
              waterFlow: reading.water_rate_cm_per_min,
            }),
          }),
        )

        if (!cancelled && points.length > 0) setSensorHistory(points)
      } catch {
        // Keep the graph stable when the sensor service is unavailable.
      }
    }

    async function loadFloodRisk() {
      try {
        const result = await apiGet<FloodRiskLiveResponse>("/flood-risk/live")
        if (!result.success) {
          throw new Error("The prediction service returned an unsuccessful response.")
        }
        if (!cancelled) {
          setFloodRiskData(result)
          setFloodRiskError(null)
        }
      } catch {
        if (!cancelled) {
          setFloodRiskData(null)
          setFloodRiskError("Unable to retrieve live flood-risk data.")
        }
      } finally {
        if (!cancelled) setFloodRiskLoading(false)
      }
    }

    function refreshDashboardData() {
      void loadSensorData()
      void loadFloodRisk()
    }

    refreshDashboardData()
    const timer = window.setInterval(refreshDashboardData, 15000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  const graph = useMemo(() => SENSOR_META[selectedSensor], [selectedSensor])

  const risk = floodRiskData?.flood_prediction.risk_label
  const riskColor =
    risk === "CRITICAL"
      ? "text-red-600"
      : risk === "HIGH"
        ? "text-orange-600"
        : risk === "MEDIUM"
          ? "text-yellow-600"
          : risk === "LOW"
            ? "text-green-600"
            : "text-slate-500"

  const riskBadge =
    risk === "CRITICAL"
      ? "bg-red-50 text-red-600"
      : risk === "HIGH"
        ? "bg-orange-50 text-orange-600"
        : risk === "MEDIUM"
          ? "bg-yellow-50 text-yellow-600"
          : risk === "LOW"
            ? "bg-green-50 text-green-600"
            : "bg-slate-100 text-slate-500"

    const predictionTimestamp = floodRiskData?.thingspeak.created_at
    const predictionAge = predictionTimestamp && currentTime
      ? formatElapsedTime(predictionTimestamp, currentTime)
      : "Timestamp unavailable"

  const broadcastAlert = () => {
    window.alert("Broadcast alert initiated.")
  }

  return (
    <main className="min-h-screen bg-[#eef4f8] text-slate-800">
      <div className="mx-auto w-full max-w-375 px-4 py-5 sm:px-6 lg:px-8">

        {/* PAGE TITLE */}
        <section className="mb-5 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="min-w-0">
              <div className="text-[10px] font-bold tracking-[0.2em] text-blue-600">
                FLOODGUARD
              </div>
              <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-800 sm:text-[28px]">
                Emergency Response Dashboard
              </h1>
              <p className="mt-1 max-w-2xl text-sm leading-5 text-slate-500">
                Current flood-risk predictions, live sensor conditions,
                emergency requests and shelter infrastructure.
              </p>
            </div>

            <button
              onClick={broadcastAlert}
              className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700"
            >
              <Megaphone className="h-4 w-4" />
              Broadcast Alert
            </button>
          </div>
        </section>

        {/* HEATMAP + MODEL */}
        <section className="mb-5 grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1.7fr)_minmax(320px,0.9fr)]">

          {/* HEATMAP */}
          <section className="min-w-0 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <div className="text-[10px] font-bold tracking-[0.16em] text-slate-400">
                  LIVE FLOOD MAP
                </div>
                <h2 className="mt-1 text-lg font-bold text-slate-700">
                  Flood Risk Heatmap
                </h2>
              </div>
              <button className="rounded-lg border border-slate-200 px-3 py-2 text-xs font-medium text-slate-600 hover:bg-slate-50">
                Expand
              </button>
            </div>

            <div className="relative h-97.5 overflow-hidden rounded-lg border border-slate-200 bg-[#e6eef5]">

              {/* map background */}
              <div className="absolute inset-0">
                <div className="absolute left-[14%] top-0 h-full w-px rotate-[4deg] bg-slate-300/50" />
                <div className="absolute left-[31%] top-0 h-full w-px rotate-18 bg-slate-300/40" />
                <div className="absolute left-[51%] top-0 h-full w-4.5 skew-x-[-10deg] bg-blue-400/40" />
                <div className="absolute left-[72%] top-0 h-full w-px -rotate-12 bg-slate-300/40" />
                <div className="absolute left-0 top-[27%] h-px w-full rotate-3 bg-slate-300/35" />
                <div className="absolute left-0 top-[58%] h-px w-full rotate-[-5deg] bg-slate-300/35" />
              </div>

              {/* heat zones */}
              <HeatZone left="18%" top="65%" size="medium" risk="LOW" />
              <HeatZone left="42%" top="45%" size="large" risk="MODERATE" />
              <HeatZone left="67%" top="30%" size="large" risk="CRITICAL" />
              <HeatZone left="74%" top="18%" size="medium" risk="HIGH" />
              <HeatZone left="55%" top="52%" size="medium" risk="CRITICAL" />

              <MapPoint left="20%" top="70%" label="Guwahati" />
              <MapPoint left="48%" top="48%" label="Jorhat" />
              <MapPoint left="58%" top="55%" label="Sivasagar" />
              <MapPoint left="72%" top="38%" label="Dibrugarh" />
              <MapPoint left="80%" top="25%" label="Tinsukia" />

              <div className="absolute bottom-3 left-3 rounded-md bg-white px-3 py-2 text-[10px] font-semibold text-slate-600 shadow-sm">
                ASSAM · LIVE RISK ZONES
              </div>

              <div className="absolute bottom-3 right-3 flex flex-wrap justify-end gap-x-3 gap-y-1 rounded-md bg-white px-3 py-2 text-[10px] shadow-sm">
                <LegendItem label="Low" className="bg-green-500" />
                <LegendItem label="Moderate" className="bg-yellow-400" />
                <LegendItem label="High" className="bg-orange-500" />
                <LegendItem label="Critical" className="bg-red-600" />
              </div>
            </div>
          </section>

          {/* MODEL 1 */}
          <section className="min-w-0 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
                <Zap className="h-5 w-5" />
              </div>
              <div>
                <div className="text-[10px] font-bold tracking-[0.16em] text-slate-400">
                  MODEL 1
                </div>
                <h2 className="text-lg font-bold text-slate-700">
                  Flood Risk Prediction
                </h2>
              </div>
            </div>

            <div className="flex justify-center py-7">
              <div className={`flex h-40 w-40 flex-col items-center justify-center rounded-full border-10 border-slate-200 bg-orange-50 ${riskColor}`}>
                {floodRiskLoading ? (
                  <div className="px-4 text-center text-sm font-semibold leading-5 text-slate-500">
                    Loading live prediction...
                  </div>
                ) : floodRiskData ? (
                  <div className="px-4 text-center text-xl font-bold leading-6">
                    {risk} RISK
                  </div>
                ) : (
                  <div className="px-4 text-center text-sm font-semibold leading-5 text-slate-500">
                    Prediction unavailable
                  </div>
                )}
              </div>
            </div>

            {floodRiskError && (
              <p role="status" className="-mt-4 mb-4 text-center text-xs text-red-600">
                {floodRiskError}
              </p>
            )}

            <div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-500">Model confidence</span>
                <span className="font-bold text-slate-700">
                  {floodRiskData
                    ? `${(floodRiskData.flood_prediction.confidence * 100).toFixed(2)}%`
                    : "--"}
                </span>
              </div>
              <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full rounded-full bg-blue-600"
                  style={{
                    width: `${(floodRiskData?.flood_prediction.confidence ?? 0) * 100}%`,
                  }}
                />
              </div>
            </div>

            <div className="mt-5 divide-y divide-slate-100 rounded-lg border border-slate-100">
              <InfoRow
                label="Prediction"
                value={risk ?? (floodRiskLoading ? "Loading..." : "Unavailable")}
                valueClass={riskBadge}
              />
              <InfoRow label="Model" value="Model 1" />
              <InfoRow
                label="ThingSpeak data age"
                value={predictionAge}
              />
            </div>

            <div className="mt-5 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-500">
              Prediction uses the latest available ThingSpeak reading and the
              configured rainfall and elevation inputs.
            </div>
          </section>
        </section>

        {/* SENSOR CARDS */}
        <section className="mb-5">
          <div className="mb-3">
            <div className="text-[10px] font-bold tracking-[0.16em] text-slate-400">
              SENSOR NETWORK
            </div>
            <h2 className="mt-1 text-lg font-bold text-slate-700">
              Live Monitoring
            </h2>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
            <SensorCard icon={<Droplet />} title="Water Level" value={SENSOR_DATA.waterLevel} unit="m" status="HIGH" statusClass="bg-orange-50 text-orange-600" />
            <SensorCard icon={<Wind />} title="Rainfall" value={SENSOR_DATA.rainfall} unit="mm" status="WARNING" statusClass="bg-orange-50 text-orange-600" />
            <SensorCard icon={<Leaf />} title="Soil Moisture" value={SENSOR_DATA.soilMoisture} unit="%" status="HIGH" statusClass="bg-blue-50 text-blue-600" />
            <SensorCard icon={<Thermometer />} title="Temperature" value={SENSOR_DATA.temperature} unit="°C" status="NORMAL" statusClass="bg-slate-100 text-slate-600" />
            <SensorCard icon={<Gauge />} title="Movement / Tilt" value={SENSOR_DATA.movement} unit="g" status="NORMAL" statusClass="bg-green-50 text-green-600" />
            <SensorCard icon={<Activity />} title="Water Flow" value={SENSOR_DATA.waterFlow} unit="m/s" status="CRITICAL" statusClass="bg-red-50 text-red-600" />
          </div>
        </section>

        {/* SENSOR TREND */}
        <section className="mb-5 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <div className="text-[10px] font-bold tracking-[0.16em] text-slate-400">
                SENSOR ANALYTICS
              </div>
              <h2 className="mt-1 text-lg font-bold text-slate-700">
                Live Sensor Trends
              </h2>
            </div>

            <select
              value={selectedSensor}
              onChange={(e) => setSelectedSensor(e.target.value as SensorKey)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-600 outline-none focus:border-blue-500 sm:w-48"
            >
              <option value="waterLevel">Water Level</option>
              <option value="waterFlow">Water Flow</option>
              <option value="soilMoisture">Soil Moisture</option>
              <option value="temperature">Temperature</option>
              <option value="movement">Ground Movement</option>
            </select>
          </div>

          <div className="mt-4 h-77.5 w-full min-w-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sensorHistory} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="time" tick={{ fontSize: 11, fill: "#64748b" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: "#64748b" }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#fff",
                    border: "1px solid #e2e8f0",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey={selectedSensor}
                  stroke="#2563eb"
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="flex flex-col gap-1 border-t border-slate-100 pt-3 text-[10px] text-slate-400 sm:flex-row sm:items-center sm:justify-between">
            <span>
              Showing {graph.title} readings · Unit: {graph.unit}
            </span>
            <span className="inline-flex items-center gap-1">
              <Radio className="h-3 w-3 text-green-600" />
              Updated {currentTime?.toLocaleTimeString("en-IN") ?? "--:--:--"}
            </span>
          </div>
        </section>

        {/* Keep this only if you want the infrastructure footer section. */}
        <section className="mt-5 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <div className="text-[10px] font-bold tracking-[0.16em] text-slate-400">
                SYSTEM STATUS
              </div>
              <h2 className="mt-1 text-lg font-bold text-slate-700">
                FloodGuard Infrastructure
              </h2>
            </div>
            <div className="inline-flex items-center gap-2 text-xs font-medium text-green-600">
              <span className="h-2 w-2 rounded-full bg-green-500" />
              All systems operational
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <SystemStatus title="ML Model" status="Operational" />
            <SystemStatus title="API" status="Connected" />
            <SystemStatus title="Sensor Network" status="Operational" />
            <SystemStatus title="LoRa" status="Connected" />
          </div>
        </section>
      </div>
    </main>
  )
}

function SensorCard({
  icon,
  title,
  value,
  unit,
  status,
  statusClass,
}: {
  icon: React.ReactNode
  title: string
  value: number
  unit: string
  status: string
  statusClass: string
}) {
  return (
    <div className="min-w-0 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex min-h-9.5 items-start justify-between gap-2">
        <div className="min-w-0 text-xs font-medium leading-4 text-slate-500">{title}</div>
        <span className={`shrink-0 rounded px-2 py-1 text-[8px] font-bold ${statusClass}`}>{status}</span>
      </div>

      <div className="mt-3 flex items-end gap-1">
        <span className="text-2xl font-bold text-slate-700">{value}</span>
        <span className="mb-1 text-xs text-slate-400">{unit}</span>
      </div>

      <div className="mt-3 flex items-center gap-1 text-[9px] font-medium text-green-600">
        <Radio className="h-3 w-3" />
        LIVE · 12 SEC AGO
      </div>
    </div>
  )
}

function InfoRow({
  label,
  value,
  valueClass = "",
}: {
  label: string
  value: string
  valueClass?: string
}) {
  return (
    <div className="flex items-center justify-between gap-3 px-3 py-2.5 text-xs">
      <span className="text-slate-400">{label}</span>
      <span className={`rounded px-1.5 py-0.5 font-semibold text-slate-700 ${valueClass}`}>{value}</span>
    </div>
  )
}

function SystemStatus({ title, status }: { title: string; status: string }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-green-100 bg-green-50 p-3">
      <div>
        <div className="text-xs font-semibold text-slate-700">{title}</div>
        <div className="mt-1 text-[10px] text-green-600">{status}</div>
      </div>
      <CheckCircle2 className="h-5 w-5 text-green-600" />
    </div>
  )
}

function formatElapsedTime(timestamp: string, now: Date) {
  const timestampMs = Date.parse(timestamp)
  if (!Number.isFinite(timestampMs)) return "Timestamp unavailable"

  const elapsedSeconds = Math.max(0, Math.floor((now.getTime() - timestampMs) / 1000))
  if (elapsedSeconds < 60) return `${elapsedSeconds} sec ago`

  const elapsedMinutes = Math.floor(elapsedSeconds / 60)
  if (elapsedMinutes < 60) return `${elapsedMinutes} min ago`

  const elapsedHours = Math.floor(elapsedMinutes / 60)
  if (elapsedHours < 24) return `${elapsedHours} hr ago`

  const elapsedDays = Math.floor(elapsedHours / 24)
  return `${elapsedDays} day${elapsedDays === 1 ? "" : "s"} ago`
}

function LegendItem({ label, className }: { label: string; className: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className={`h-2 w-2 rounded-full ${className}`} />
      <span className="text-slate-500">{label}</span>
    </span>
  )
}

function HeatZone({
  left,
  top,
  size,
  risk,
}: {
  left: string
  top: string
  size: "small" | "medium" | "large"
  risk: "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
}) {
  const sizes = {
    small: "h-12 w-12",
    medium: "h-20 w-20",
    large: "h-32 w-32",
  }

  const colors = {
    LOW: "bg-green-400/25",
    MODERATE: "bg-yellow-400/30",
    HIGH: "bg-orange-500/35",
    CRITICAL: "bg-red-500/45",
  }

  return (
    <div
      className={`absolute rounded-full blur-2xl ${sizes[size]} ${colors[risk]}`}
      style={{ left, top }}
    />
  )
}

function MapPoint({
  left,
  top,
  label,
}: {
  left: string
  top: string
  label: string
}) {
  return (
    <div className="absolute" style={{ left, top }}>
      <div className="relative">
        <div className="h-3 w-3 rounded-full border-2 border-white bg-red-600 shadow-sm" />
        <div className="absolute left-4 -top-1.25 whitespace-nowrap rounded bg-white px-2 py-1 text-[9px] font-semibold text-slate-600 shadow-sm">
          {label}
        </div>
      </div>
    </div>
  )
}
