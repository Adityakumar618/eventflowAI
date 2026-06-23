import { useState } from 'react'
import { Loader2, Map, Navigation, Shield, Zap } from 'lucide-react'
import GlassCard from '../components/GlassCard'
import FormSelect from '../components/FormSelect'
import MapplsEventMap from '../components/MapplsEventMap'
import { api } from '../api'
import { ACCENT, GRADIENT, SEMANTIC } from '../theme/palette'

const DEFAULT = {
  event_cause: 'water_logging',
  corridor: 'Hebbal - Silk Board',
  zone: 'Central',
  hour: 18,
  requires_road_closure: false,
  lat: 13.035,
  lon: 77.597,
}

const CAUSES = [
  'water_logging', 'tree_fall', 'accident', 'vehicle_breakdown',
  'construction', 'congestion', 'pot_holes', 'road_conditions', 'others',
]

const ZONES = ['North', 'South', 'East', 'West', 'Central']

// ── Survival chart ───────────────────────────────────────────────────────────
function SurvivalChart({ survival }) {
  if (!survival?.available) return <p className="text-storm-100/50 text-sm">No historical survival data for this cause.</p>
  const { timeline, survival: surv, p25_hours, p50_hours, p10_hours } = survival
  const w = 400; const h = 160; const pad = 24
  const pts = timeline.map((t, i) => {
    const x = pad + (t / (survival.x_max || 48)) * (w - pad * 2)
    const y = h - pad - surv[i] * (h - pad * 2)
    return `${x},${y}`
  }).join(' ')
  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-40">
        <polyline fill="none" stroke={ACCENT.gold} strokeWidth="2" points={pts} />
        <polygon fill="rgba(214,154,45,0.08)" points={`${pad},${h - pad} ${pts} ${w - pad},${h - pad}`} />
      </svg>
      <div className="flex flex-wrap gap-3 text-xs text-storm-100/60 mt-2">
        {p25_hours && <span>Q25: {p25_hours}h</span>}
        {p50_hours && <span>Median: {p50_hours}h</span>}
        {p10_hours && <span>P10: {p10_hours}h</span>}
        {survival.censor_rate > 30 && <span className="text-accent-amber">⚠ {survival.censor_rate}% censoring</span>}
      </div>
    </div>
  )
}

// ── Prescriptive Panel ───────────────────────────────────────────────────────
function PrescriptivePanel({ presc }) {
  if (!presc) return null
  const mp = presc.manpower || {}
  const bd = presc.barricade_diversion || {}
  const wi = presc.what_if || {}
  return (
    <GlassCard className="p-5 border-l-4" style={{ borderLeftColor: ACCENT.success }}>
      <h3 className="text-storm-100 font-semibold mb-3 flex items-center gap-2">
        <Shield size={18} className="text-accent-success" /> Prescriptive Recommendations
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        <div className="space-y-1">
          <p className="text-storm-100/50 text-xs uppercase">Manpower</p>
          <p className="text-2xl font-black text-storm-300">{mp.total_deployed ?? '—'} <span className="text-sm font-normal text-storm-100/50">officers</span></p>
          <p className="text-storm-100/60 text-xs">{mp.station || 'Nearest station'}</p>
          {mp.narrative && <p className="text-storm-100/50 text-xs mt-1">{mp.narrative}</p>}
        </div>
        <div className="space-y-1">
          <p className="text-storm-100/50 text-xs uppercase">Barricade Level</p>
          <p className="text-2xl font-black" style={{ color: bd.barricade_level === 'FULL' ? ACCENT.terracotta : bd.barricade_level === 'PARTIAL' ? ACCENT.amber : ACCENT.success }}>
            {bd.barricade_level ?? '—'}
          </p>
          {bd.diversion_routes?.length > 0 && (
            <div className="mt-1 space-y-0.5">
              <p className="text-storm-100/40 text-xs">Diversion routes:</p>
              {bd.diversion_routes.slice(0, 2).map((r, i) => (
                <p key={i} className="text-storm-100/60 text-xs flex items-center gap-1">
                  <Navigation size={9} /> {typeof r === 'string' ? r : r.route || r.description || JSON.stringify(r)}
                </p>
              ))}
            </div>
          )}
        </div>
        <div className="space-y-1">
          <p className="text-storm-100/50 text-xs uppercase">What-if Improvement</p>
          <p className="text-2xl font-black text-accent-success">
            {wi.delay_reduction_pct != null ? `-${wi.delay_reduction_pct.toFixed(0)}%` : wi.estimated_reduction_hrs != null ? `-${wi.estimated_reduction_hrs.toFixed(1)}h` : '—'}
          </p>
          <p className="text-storm-100/60 text-xs">{wi.narrative || wi.summary || 'vs. no-action baseline'}</p>
          {presc.confidence && (
            <p className="text-storm-100/40 text-xs mt-1">Confidence: {(presc.confidence * 100).toFixed(0)}%</p>
          )}
        </div>
      </div>
      {presc.rationale && (
        <p className="mt-3 text-storm-100/50 text-xs border-t border-white/[0.06] pt-3">{presc.rationale}</p>
      )}
    </GlassCard>
  )
}

// ── Main View ────────────────────────────────────────────────────────────────
export default function NewEventView() {
  const [form, setForm] = useState(DEFAULT)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [triage, setTriage] = useState(null)
  const [presc, setPresc] = useState(null)
  const [error, setError] = useState(null)

  const update = (k, v) => setForm((f) => ({ ...f, [k]: v }))

  const run = async () => {
    setLoading(true); setError(null); setPresc(null)
    try {
      const pred = await api.predictEvent(form)
      setResult(pred)
      const tri = await api.triage({ new_event: form, zone: form.zone, capacity: 14 })
      setTriage(tri)
      try {
        const prescResult = await api.prescriptive({
          event: {
            event_id: `event_${Date.now()}`,
            event_cause: form.event_cause,
            corridor: form.corridor,
            zone: form.zone,
            police_station: 'Hebbal',
            hour: form.hour,
            lat: form.lat,
            lon: form.lon,
            requires_road_closure: form.requires_road_closure,
            predicted_hours: pred.ml_prediction?.predicted_hours,
            p35_hours: pred.ml_prediction?.p10_hours,
            p50_hours: pred.ml_prediction?.predicted_hours,
            p65_hours: pred.ml_prediction?.p90_hours,
            impact_score: (pred.stis?.stis || 5) * (pred.ml_prediction?.predicted_hours || 2),
            regime: (pred.ml_prediction?.predicted_hours || 2) < 2 ? 'short' : (pred.ml_prediction?.predicted_hours || 2) < 6 ? 'medium' : 'long',
            active_overlap: 2,
          }
        })
        setPresc(prescResult)
      } catch (_) { /* prescriptive is optional */ }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-storm-100 flex items-center gap-2">
          <Zap className="text-storm-300" size={24} /> New Event Intake
        </h2>
        <p className="text-storm-muted text-sm mt-1">Survival curves · STIS · Economic impact · Officer allocation · Live map</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <GlassCard className="p-5 lg:col-span-1 space-y-4">
          <h3 className="text-storm-100 font-semibold">Event Details</h3>
          <label className="block text-xs text-storm-100/60">Cause</label>
          <FormSelect value={form.event_cause} onChange={(e) => update('event_cause', e.target.value)}>
            {CAUSES.map((c) => (
              <option key={c} value={c}>{c.replace(/_/g, ' ')}</option>
            ))}
          </FormSelect>
          <label className="block text-xs text-storm-100/60">Corridor</label>
          <input value={form.corridor} onChange={(e) => update('corridor', e.target.value)} className="storm-input w-full rounded-lg py-2 px-3" />
          <label className="block text-xs text-storm-100/60">Zone</label>
          <FormSelect value={form.zone} onChange={(e) => update('zone', e.target.value)}>
            {ZONES.map((z) => <option key={z} value={z}>{z}</option>)}
          </FormSelect>
          <label className="block text-xs text-storm-100/60">Hour: {form.hour}:00</label>
          <input type="range" min={0} max={23} value={form.hour} onChange={(e) => update('hour', +e.target.value)} className="w-full accent-storm-300" />
          <div className="grid grid-cols-2 gap-2 text-xs text-storm-100/50">
            <div>
              <label className="block mb-1">Lat</label>
              <input type="number" step="0.001" value={form.lat} onChange={(e) => update('lat', +e.target.value)} className="storm-input w-full rounded-lg py-1 px-2 text-xs" />
            </div>
            <div>
              <label className="block mb-1">Lon</label>
              <input type="number" step="0.001" value={form.lon} onChange={(e) => update('lon', +e.target.value)} className="storm-input w-full rounded-lg py-1 px-2 text-xs" />
            </div>
          </div>
          <label className="flex items-center gap-2 text-sm text-storm-100">
            <input type="checkbox" checked={form.requires_road_closure} onChange={(e) => update('requires_road_closure', e.target.checked)} className="accent-storm-300" />
            Requires road closure
          </label>
          <button onClick={run} disabled={loading} className="storm-cta w-full py-3 rounded-xl flex items-center justify-center gap-2 disabled:opacity-50 transition-all" style={{ background: GRADIENT.ctaAlt }}>
            {loading ? <Loader2 className="animate-spin" size={18} /> : <Zap size={18} />}
            Generate Intelligence
          </button>
          {error && <p className="alert-rose text-sm rounded-lg px-3 py-2">{error}</p>}
        </GlassCard>

        <div className="lg:col-span-2 space-y-4">
          <GlassCard className="p-4">
            <h3 className="text-storm-100 font-semibold mb-3 flex items-center gap-2">
              <Map size={16} className="text-storm-300" /> Live Map — Bengaluru Traffic
            </h3>
            <MapplsEventMap
              lat={form.lat}
              lon={form.lon}
              eventCause={form.event_cause}
              height={340}
              onLocationChange={(newLat, newLon) => setForm((f) => ({ ...f, lat: newLat, lon: newLon }))}
            />
          </GlassCard>

          {!result ? (
            <GlassCard className="p-8 text-center text-storm-100/40">
              Fill in event details and click Generate Intelligence.
            </GlassCard>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <GlassCard className="p-5 bento-card">
                  <h3 className="text-storm-100/60 text-xs uppercase mb-2">GridGuard V12 Prediction</h3>
                  <div className="text-4xl font-black text-storm-300">{result.ml_prediction.predicted_hours}h</div>
                  <p className="text-storm-100/50 text-sm mt-1">
                    P10 {result.ml_prediction.p10_hours}h · P90 {result.ml_prediction.p90_hours}h
                  </p>
                </GlassCard>
                <GlassCard className="p-5 border-l-4" style={{ borderLeftColor: result.stis.color || SEMANTIC.warning }}>
                  <h3 className="text-storm-100/60 text-xs uppercase mb-2">STIS · {result.stis.label}</h3>
                  <div className="text-4xl font-black" style={{ color: result.stis.color || ACCENT.amber }}>
                    {result.stis.stis}<span className="text-lg text-storm-100/40">/10</span>
                  </div>
                  <p className="text-storm-100/50 text-sm mt-2">{result.stis.deployment_note}</p>
                </GlassCard>
              </div>

              <PrescriptivePanel presc={presc} />

              <GlassCard className="p-5">
                <h3 className="text-storm-100 font-semibold mb-3">Survival Analysis</h3>
                <SurvivalChart survival={result.survival} />
                {result.chronic_warning && <p className="alert-amber text-sm mt-3 rounded-lg p-3">{result.chronic_warning}</p>}
              </GlassCard>

              <GlassCard className="p-5 border-l-4 border-accent-amber">
                <h3 className="text-storm-100/60 text-xs uppercase mb-2">Economic Impact</h3>
                <div className="text-3xl font-black text-accent-amber">{result.economic.cost_display}</div>
                <p className="text-storm-100/50 text-sm mt-2">
                  {result.economic.affected_commuters.toLocaleString()} commuters · {result.economic.peak_label}
                </p>
              </GlassCard>

              {triage && (
                <GlassCard className="p-5">
                  <h3 className="text-storm-100 font-semibold mb-3">Multi-Event Triage</h3>
                  {triage.triage_active ? (
                    <p className="alert-rose text-sm mb-3 rounded-lg px-3 py-2">⚠ TRIAGE ACTIVE — {triage.capacity} officers in Zone {triage.zone}</p>
                  ) : (
                    <p className="alert-emerald text-sm mb-3 rounded-lg px-3 py-2">✓ All events fully covered</p>
                  )}
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead className="text-storm-100/50 text-xs uppercase">
                        <tr><th className="py-2">Event</th><th>STIS</th><th>Needed</th><th>Assigned</th><th>Coverage</th></tr>
                      </thead>
                      <tbody>
                        {triage.rows.map((r) => (
                          <tr key={r.event} className="border-t border-white/5">
                            <td className="py-2 text-storm-100">{r.event}</td>
                            <td className="text-storm-300">{r.stis}</td>
                            <td>{r.min_needed}</td>
                            <td>{r.assigned}</td>
                            <td>{r.coverage}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </GlassCard>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}