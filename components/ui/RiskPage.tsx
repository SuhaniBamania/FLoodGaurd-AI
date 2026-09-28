"use client"

import { useEffect, useState } from "react"

type Emergency = {
  sos_id: string | number
  created_at: string
  status: string
  location: string
  latitude: number | null
  longitude: number | null

  sos_data?: {
    people?: {
      total?: string | number | null
      children?: string | number | null
      elderly?: string | number | null
      pregnant?: string | number | null
      injured?: string | number | null
      missing?: string | number | null
      deceased?: string | number | null
      mobility_impaired?: string | number | null
    }
    situation?: string
    request?: {
      type?: string
      resources?: string[]
    }
  }

  original_message?: string

  drone?: {
    available: boolean
    people_detected: number
    location: string | null
    latitude: number | null
    longitude: number | null
    distance_from_sos_km: number | null
    image_path: string | null
    confidence: number | null
    model_name: string | null
    analysis_status: string | null
  }

  flood?: {
    severity: number
    available: boolean
  }

  priority: {
    priority: string
    priority_score: number
    score_breakdown: {
      people_score: number
      vulnerability_score: number
      needs_score: number
      request_score: number
      flood_score: number
      location_score: number
    }
    inputs: {
      sos_people: number
      feature2_people: number | null
      effective_people: number
      flood_severity: number
    }
    recommended_actions: string[]
  }

  rank: number
}

type DashboardResponse = {
  success: boolean
  count: number
  match_radius_km: number
  flood_severity_used: number
  emergencies: Emergency[]
}

const API_URL =
  process.env.NEXT_PUBLIC_FLOODGAURD_API_URL || "http://127.0.0.1:8000"

function getPriorityStyle(priority: string) {
  switch (priority?.toUpperCase()) {
    case "CRITICAL":
      return {
        badge: "bg-red-50 text-red-700 border-red-200",
        dot: "bg-red-500",
        text: "text-red-700",
        bar: "bg-red-500",
      }

    case "HIGH":
      return {
        badge: "bg-orange-50 text-orange-700 border-orange-200",
        dot: "bg-orange-500",
        text: "text-orange-700",
        bar: "bg-orange-500",
      }

    case "MEDIUM":
      return {
        badge: "bg-amber-50 text-amber-700 border-amber-200",
        dot: "bg-amber-500",
        text: "text-amber-700",
        bar: "bg-amber-500",
      }

    default:
      return {
        badge: "bg-emerald-50 text-emerald-700 border-emerald-200",
        dot: "bg-emerald-500",
        text: "text-emerald-700",
        bar: "bg-emerald-500",
      }
  }
}

function getFloodLabel(severity: number) {
  if (severity >= 1) return "CRITICAL"
  if (severity >= 0.75) return "HIGH"
  if (severity >= 0.5) return "MEDIUM"
  return "LOW"
}

function formatAction(action: string) {
  return action
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function formatDate(date: string) {
  try {
    return new Date(date).toLocaleString([], {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    })
  } catch {
    return date
  }
}

export default function RiskPage() {
  const [data, setData] = useState<DashboardResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [expandedId, setExpandedId] = useState<string | null>(null)

  async function fetchDashboard() {
    try {
      setLoading(true)
      setError("")

      const response = await fetch(
        `${API_URL}/priority-dashboard?flood_severity=1.0&match_radius_km=5`,
        {
          cache: "no-store",
        }
      )

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`)
      }

      const result: DashboardResponse = await response.json()

      setData(result)
    } catch (err) {
      console.error("Priority dashboard error:", err)
      setError(
        "Unable to connect to the emergency intelligence service."
      )
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDashboard()
  }, [])

  const emergencies = data?.emergencies || []

  const criticalCount = emergencies.filter(
    (item) => item.priority.priority === "CRITICAL"
  ).length

  const highCount = emergencies.filter(
    (item) => item.priority.priority === "HIGH"
  ).length

  const totalPeople = emergencies.reduce(
    (sum, item) => sum + (item.priority.inputs.effective_people || 0),
    0
  )

  const droneLocations = emergencies.filter(
    (item) => item.drone?.available
  ).length

  return (
    <div className="space-y-6">

      {/* HEADER */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#66819D]">
            Emergency operations / priority center
          </div>

          <h1 className="text-2xl font-semibold tracking-tight text-[#163A63]">
            Emergency Priority Command
          </h1>

          <p className="mt-1 text-sm text-[#627D98]">
            Location-wise emergency intelligence ranked by urgency.
          </p>
        </div>

        <button
          onClick={fetchDashboard}
          disabled={loading}
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-[#D9E2EC] bg-white px-4 py-2.5 text-sm font-medium text-[#243B53] shadow-sm transition hover:bg-[#F5F7FA] disabled:cursor-not-allowed disabled:opacity-60"
        >
          <span className={loading ? "animate-spin" : ""}>↻</span>
          {loading ? "Refreshing..." : "Refresh intelligence"}
        </button>
      </div>


      {/* LIVE STATUS */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-emerald-200 bg-emerald-50/70 px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
          </span>

          <div>
            <div className="text-sm font-semibold text-emerald-800">
              Priority engine online
            </div>

            <div className="text-xs text-emerald-700">
              SOS + drone intelligence + flood severity
            </div>
          </div>
        </div>

        <div className="text-xs font-medium text-emerald-700">
          5 km location matching
        </div>
      </div>


      {/* SUMMARY CARDS */}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">

        <div className="rounded-xl border border-[#D9E2EC] bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-[#829AB1]">
              Active emergencies
            </span>

            <div className="rounded-lg bg-[#EFF6FF] px-2.5 py-1.5 text-lg">
              🚨
            </div>
          </div>

          <div className="mt-3 text-3xl font-bold text-[#163A63]">
            {loading ? "—" : emergencies.length}
          </div>

          <div className="mt-1 text-xs text-[#627D98]">
            Ranked locations
          </div>
        </div>


        <div className="rounded-xl border border-red-100 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-[#829AB1]">
              Critical
            </span>

            <div className="rounded-lg bg-red-50 px-2.5 py-1.5 text-lg">
              🔴
            </div>
          </div>

          <div className="mt-3 text-3xl font-bold text-red-600">
            {loading ? "—" : criticalCount}
          </div>

          <div className="mt-1 text-xs text-[#627D98]">
            Immediate response
          </div>
        </div>


        <div className="rounded-xl border border-orange-100 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-[#829AB1]">
              High priority
            </span>

            <div className="rounded-lg bg-orange-50 px-2.5 py-1.5 text-lg">
              🟠
            </div>
          </div>

          <div className="mt-3 text-3xl font-bold text-orange-600">
            {loading ? "—" : highCount}
          </div>

          <div className="mt-1 text-xs text-[#627D98]">
            Requires attention
          </div>
        </div>


        <div className="rounded-xl border border-[#D9E2EC] bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-[#829AB1]">
              People at risk
            </span>

            <div className="rounded-lg bg-[#F0FDF4] px-2.5 py-1.5 text-lg">
              👥
            </div>
          </div>

          <div className="mt-3 text-3xl font-bold text-[#163A63]">
            {loading ? "—" : totalPeople}
          </div>

          <div className="mt-1 text-xs text-[#627D98]">
            SOS + drone intelligence
          </div>
        </div>

      </div>


      {/* ERROR */}
      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4">
          <div className="font-semibold text-red-800">
            Connection error
          </div>

          <div className="mt-1 text-sm text-red-700">
            {error}
          </div>

          <button
            onClick={fetchDashboard}
            className="mt-3 rounded-lg bg-red-600 px-3 py-2 text-xs font-semibold text-white hover:bg-red-700"
          >
            Try again
          </button>
        </div>
      )}


      {/* RANKED EMERGENCIES */}
      <div className="overflow-hidden rounded-xl border border-[#D9E2EC] bg-white shadow-sm">

        <div className="border-b border-[#E6EDF3] px-5 py-5">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">

            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#66819D]">
                Live ranking
              </div>

              <h2 className="mt-1 text-lg font-semibold text-[#163A63]">
                Priority emergencies
              </h2>

              <p className="mt-1 text-sm text-[#627D98]">
                Highest priority appears first. Select an emergency for full intelligence.
              </p>
            </div>

            <div className="rounded-lg bg-[#F5F7FA] px-3 py-2 text-xs font-medium text-[#486581]">
              {droneLocations} drone-linked location
              {droneLocations === 1 ? "" : "s"}
            </div>

          </div>
        </div>


        {/* TABLE HEADER */}
        <div className="hidden border-b border-[#E6EDF3] bg-[#F8FAFC] px-5 py-3 lg:grid lg:grid-cols-[70px_1.5fr_90px_90px_100px_130px_90px] lg:items-center lg:gap-4">
          <div className="text-[10px] font-bold uppercase tracking-wider text-[#829AB1]">
            Rank
          </div>

          <div className="text-[10px] font-bold uppercase tracking-wider text-[#829AB1]">
            Location
          </div>

          <div className="text-[10px] font-bold uppercase tracking-wider text-[#829AB1]">
            SOS
          </div>

          <div className="text-[10px] font-bold uppercase tracking-wider text-[#829AB1]">
            Drone
          </div>

          <div className="text-[10px] font-bold uppercase tracking-wider text-[#829AB1]">
          Flood
          </div>

          <div className="text-[10px] font-bold uppercase tracking-wider text-[#829AB1]">
          Score 
          </div>

          <div className="text-[10px] font-bold uppercase tracking-wider text-[#829AB1]">
          Priority
          </div>
        </div>


        {/* LOADING */}
        {loading && (
          <div className="divide-y divide-[#E6EDF3]">
            {[1, 2, 3].map((item) => (
              <div
                key={item}
                className="animate-pulse px-5 py-5"
              >
                <div className="h-5 w-1/3 rounded bg-slate-100" />
                <div className="mt-3 h-4 w-2/3 rounded bg-slate-100" />
              </div>
            ))}
          </div>
        )}


        {/* EMPTY */}
        {!loading && !error && emergencies.length === 0 && (
          <div className="px-6 py-16 text-center">
            <div className="text-4xl">✓</div>

            <h3 className="mt-3 text-lg font-semibold text-[#163A63]">
              No active emergencies
            </h3>

            <p className="mt-1 text-sm text-[#627D98]">
              The priority engine has no emergency requests to rank.
            </p>
          </div>
        )}


        {/* EMERGENCY ROWS */}
        {!loading && emergencies.map((item) => {
          const style = getPriorityStyle(item.priority.priority)
          const rowKey = `${item.sos_id}-${item.rank}`
          const isExpanded = expandedId === rowKey

          const sosPeople =
            item.priority.inputs.sos_people ?? 0

          const dronePeople =
            item.drone?.available
              ? item.drone.people_detected
              : 0

          const floodSeverity =
            item.flood?.severity ?? 0

          return (
            <div
              key={rowKey}
              className="border-b border-[#E6EDF3] last:border-b-0"
            >

              {/* MAIN ROW */}
              <button
                onClick={() =>
                    setExpandedId(isExpanded ? null : rowKey)
                }
                className="w-full px-5 py-5 text-left transition hover:bg-[#F8FAFC]"
              >

                {/* DESKTOP */}
<div className="hidden lg:grid lg:grid-cols-[70px_1.5fr_90px_90px_100px_90px_130px] lg:items-center lg:gap-4">

{/* RANK */}
<div>
  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#EFF6FF] text-sm font-bold text-[#2563EB]">
    #{item.rank}
  </div>
</div>


{/* LOCATION */}
<div className="min-w-0">
  <div className="truncate text-sm font-semibold text-[#163A63]">
    {item.location || "Unknown location"}
  </div>

  <div className="mt-1 text-xs text-[#829AB1]">
    SOS #{item.sos_id} · {formatDate(item.created_at)}
  </div>
</div>


{/* SOS PEOPLE */}
<div>
  <div className="text-base font-bold text-[#243B53]">
    {sosPeople}
  </div>

  <div className="text-[10px] uppercase tracking-wider text-[#829AB1]">
    reported
  </div>
</div>


{/* DRONE PEOPLE */}
<div>
  {item.drone?.available ? (
    <>
      <div className="text-base font-bold text-[#243B53]">
        {dronePeople}
      </div>

      <div className="text-[10px] uppercase tracking-wider text-emerald-600">
        detected
      </div>
    </>
  ) : (
    <span className="text-sm text-[#9FB3C8]">
      —
    </span>
  )}
</div>


{/* FLOOD SEVERITY */}
<div>
  <div className="text-sm font-semibold text-[#243B53]">
    {getFloodLabel(floodSeverity)}
  </div>

  <div className="mt-0.5 text-[10px] text-[#829AB1]">
    {Math.round(floodSeverity * 100)}% severity
  </div>
</div>


{/* FINAL PRIORITY SCORE */}
<div>
  <div className={`text-xl font-bold ${style.text}`}>
    {item.priority.priority_score}
  </div>

  <div className="mt-1 h-1.5 w-16 overflow-hidden rounded-full bg-[#E6EDF3]">
    <div
      className={`h-full rounded-full ${style.bar}`}
      style={{
        width: `${Math.min(
          item.priority.priority_score,
          100
        )}%`,
      }}
    />
  </div>
</div>


{/* PRIORITY */}
<div>
  <span
    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide ${style.badge}`}
  >
    <span
      className={`h-1.5 w-1.5 rounded-full ${style.dot}`}
    />
    {item.priority.priority}
  </span>
</div>

</div>

                {/* MOBILE */}
                <div className="lg:hidden">

                  <div className="flex items-start justify-between gap-4">

                    <div className="flex min-w-0 items-center gap-3">

                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#EFF6FF] text-xs font-bold text-[#2563EB]">
                        #{item.rank}
                      </div>

                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold text-[#163A63]">
                          {item.location || "Unknown location"}
                        </div>

                        <div className="mt-1 text-xs text-[#829AB1]">
                          SOS #{item.sos_id}
                        </div>
                      </div>

                    </div>

                    <div className="text-right">
                      <div className={`text-2xl font-bold ${style.text}`}>
                        {item.priority.priority_score}
                      </div>

                      <span
                        className={`mt-1 inline-flex rounded-full border px-2 py-0.5 text-[9px] font-bold ${style.badge}`}
                      >
                        {item.priority.priority}
                      </span>
                    </div>

                  </div>


                  <div className="mt-4 grid grid-cols-3 gap-2">

                    <div className="rounded-lg bg-[#F8FAFC] p-3">
                      <div className="text-[9px] font-bold uppercase tracking-wider text-[#829AB1]">
                        SOS
                      </div>

                      <div className="mt-1 text-sm font-bold text-[#243B53]">
                        {sosPeople} people
                      </div>
                    </div>


                    <div className="rounded-lg bg-[#F8FAFC] p-3">
                      <div className="text-[9px] font-bold uppercase tracking-wider text-[#829AB1]">
                        Drone
                      </div>

                      <div className="mt-1 text-sm font-bold text-[#243B53]">
                        {item.drone?.available
                          ? `${dronePeople} detected`
                          : "Unavailable"}
                      </div>
                    </div>


                    <div className="rounded-lg bg-[#F8FAFC] p-3">
                      <div className="text-[9px] font-bold uppercase tracking-wider text-[#829AB1]">
                        Flood
                      </div>

                      <div className="mt-1 text-sm font-bold text-[#243B53]">
                        {getFloodLabel(floodSeverity)}
                      </div>
                    </div>

                  </div>

                </div>

              </button>


              {/* EXPANDED DETAILS */}
              {isExpanded && (
                <div className="bg-[#F8FAFC] px-5 pb-6">

                  <div className="rounded-xl border border-[#D9E2EC] bg-white p-5">

                    <div className="grid gap-6 xl:grid-cols-3">

                      {/* SOS */}
                      <div>
                        <div className="mb-3 text-[10px] font-bold uppercase tracking-[0.16em] text-[#829AB1]">
                          SOS intelligence
                        </div>

                        <div className="space-y-3">

                          <div>
                            <div className="text-xs text-[#829AB1]">
                              Location
                            </div>

                            <div className="mt-1 text-sm font-semibold text-[#243B53]">
                              {item.location || "Unknown location"}
                            </div>

                            {(item.latitude != null ||
                              item.longitude != null) && (
                              <div className="mt-2 rounded-lg border border-[#E6EDF3] bg-[#F8FAFC] px-3 py-2">
                                {item.latitude != null && (
                                  <div className="text-xs text-[#627D98]">
                                    Latitude: {item.latitude}
                                  </div>
                                )}
                                {item.longitude != null && (
                                  <div className="mt-1 text-xs text-[#627D98]">
                                    Longitude: {item.longitude}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>

                          <div>
                            <div className="text-xs text-[#829AB1]">
                              Situation
                            </div>

                            <div className="mt-1 text-sm leading-6 text-[#243B53]">
                              {item.sos_data?.situation ||
                                item.original_message ||
                                "No situation details available."}
                            </div>
                          </div>


                          <div className="grid grid-cols-2 gap-2">

                            <div className="rounded-lg bg-[#F8FAFC] p-3">
                              <div className="text-[9px] uppercase tracking-wider text-[#829AB1]">
                                SOS people
                              </div>

                              <div className="mt-1 text-lg font-bold text-[#163A63]">
                                {sosPeople}
                              </div>
                            </div>

                            <div className="rounded-lg bg-[#F8FAFC] p-3">
                              <div className="text-[9px] uppercase tracking-wider text-[#829AB1]">
                                Request
                              </div>

                              <div className="mt-1 text-sm font-bold text-[#163A63]">
                                {item.sos_data?.request?.type || "—"}
                              </div>
                            </div>

                          </div>

                        </div>
                      </div>


                      {/* DRONE + FLOOD */}
                      <div>
                        <div className="mb-3 text-[10px] font-bold uppercase tracking-[0.16em] text-[#829AB1]">
                          Situational intelligence
                        </div>

                        <div className="space-y-3">

                          <div className="rounded-lg border border-[#E6EDF3] p-3">

                            <div className="flex items-center justify-between">
                              <span className="text-xs text-[#627D98]">
                                Drone detection
                              </span>

                              <span
                                className={`text-xs font-bold ${
                                  item.drone?.available
                                    ? "text-emerald-600"
                                    : "text-[#9FB3C8]"
                                }`}
                              >
                                {item.drone?.available
                                  ? "AVAILABLE"
                                  : "UNAVAILABLE"}
                              </span>
                            </div>

                            <div className="mt-2 text-lg font-bold text-[#163A63]">
                              {item.drone?.available
                                ? `${dronePeople} people detected`
                                : "No matching drone"}
                            </div>

                            {item.drone?.available &&
                              item.drone.confidence !== null && (
                                <div className="mt-1 text-xs text-[#829AB1]">
                                  Average confidence:{" "}
                                  {(
                                    item.drone.confidence * 100
                                  ).toFixed(1)}
                                  %
                                </div>
                              )}

                          </div>


                          <div className="rounded-lg border border-[#E6EDF3] p-3">

                            <div className="flex items-center justify-between">
                              <span className="text-xs text-[#627D98]">
                                Flood severity
                              </span>

                              <span className="text-xs font-bold text-red-600">
                                {getFloodLabel(floodSeverity)}
                              </span>
                            </div>

                            <div className="mt-2 h-2 overflow-hidden rounded-full bg-[#E6EDF3]">
                              <div
                                className="h-full rounded-full bg-red-500"
                                style={{
                                  width: `${Math.min(
                                    floodSeverity * 100,
                                    100
                                  )}%`,
                                }}
                              />
                            </div>

                            <div className="mt-1 text-xs text-[#829AB1]">
                              Severity:{" "}
                              {Math.round(floodSeverity * 100)}%
                            </div>

                          </div>

                        </div>
                      </div>


                      {/* PRIORITY */}
                      <div>
                        <div className="mb-3 text-[10px] font-bold uppercase tracking-[0.16em] text-[#829AB1]">
                          Priority calculation
                        </div>

                        <div className="rounded-xl border border-[#E6EDF3] p-4">

                          <div className="flex items-end justify-between">
                            <div>
                              <div className="text-xs text-[#829AB1]">
                                Final score
                              </div>

                              <div
                                className={`mt-1 text-4xl font-bold ${style.text}`}
                              >
                                {item.priority.priority_score}
                              </div>
                            </div>

                            <span
                              className={`rounded-full border px-2.5 py-1 text-[10px] font-bold ${style.badge}`}
                            >
                              {item.priority.priority}
                            </span>
                          </div>


                          <div className="mt-4 space-y-2">

                            {[
                              ["People", item.priority.score_breakdown.people_score],
                              ["Vulnerability", item.priority.score_breakdown.vulnerability_score],
                              ["Needs", item.priority.score_breakdown.needs_score],
                              ["Request", item.priority.score_breakdown.request_score],
                              ["Flood", item.priority.score_breakdown.flood_score],
                              ["Location", item.priority.score_breakdown.location_score],
                            ].map(([label, value]) => (
                              <div
                                key={String(label)}
                                className="flex items-center justify-between text-xs"
                              >
                                <span className="text-[#627D98]">
                                  {label}
                                </span>

                                <span className="font-bold text-[#243B53]">
                                  +{value}
                                </span>
                              </div>
                            ))}

                          </div>

                        </div>
                      </div>

                    </div>


                    {/* ACTIONS */}
                    <div className="mt-6 border-t border-[#E6EDF3] pt-5">

                      <div className="text-[10px] font-bold uppercase tracking-[0.16em] text-[#829AB1]">
                        Recommended response
                      </div>

                      <div className="mt-3 flex flex-wrap gap-2">

                        {(item.priority.recommended_actions || []).map(
                          (action) => (
                            <span
                              key={action}
                              className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-semibold text-blue-700"
                            >
                              {formatAction(action)}
                            </span>
                          )
                        )}

                      </div>

                    </div>


                    {/* ORIGINAL MESSAGE */}
                    {item.original_message && (
                      <div className="mt-5 rounded-lg bg-[#F8FAFC] p-4">

                        <div className="text-[10px] font-bold uppercase tracking-[0.16em] text-[#829AB1]">
                          Original SOS message
                        </div>

                        <p className="mt-2 text-sm leading-6 text-[#486581]">
                          "{item.original_message}"
                        </p>

                      </div>
                    )}

                  </div>

                </div>
              )}

            </div>
          )
        })}

      </div>


      {/* FOOTER INFO */}
      <div className="flex flex-col gap-2 rounded-xl border border-[#D9E2EC] bg-white px-5 py-4 text-xs text-[#627D98] sm:flex-row sm:items-center sm:justify-between">
        <span>
          Priority is calculated using SOS vulnerability, detected people,
          requested assistance, flood severity and location intelligence.
        </span>

        <span className="font-medium text-[#486581]">
          Ranked highest → lowest
        </span>
      </div>

    </div>
  )
}