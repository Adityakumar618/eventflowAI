import { useEffect, useState } from 'react'
import { BarChart3, Loader2 } from 'lucide-react'
import GlassCard from '../components/GlassCard'
import { api } from '../api'
import { ACCENT, WORKFLOW_ACCENTS } from '../theme/palette'

export default function AuditView() {
  const [stations, setStations] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [cascades, setCascades] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.stations(), api.metrics(), api.cascades()])
      .then(([s, m, c]) => {
        setStations(s)
        setMetrics(m)
        setCascades(c)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return <div className="flex justify-center py-20"><Loader2 className="animate-spin text-storm-300" size={32} /></div>
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-storm-100 flex items-center gap-2">
          <BarChart3 className="text-storm-300" size={24} /> Post-Event Audit
        </h2>
        <p className="text-storm-100/50 text-sm mt-1">Station efficiency, validation metrics & cascade analysis.</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {(metrics?.metrics || []).map((m, i) => {
          const accent = WORKFLOW_ACCENTS[i % WORKFLOW_ACCENTS.length]
          return (
            <GlassCard key={m.title} className="p-4 text-center bento-card !bg-white/[0.04] hover:bg-white/[0.06] transition-colors">
              <div className="text-2xl font-black" style={{ color: accent.hex }}>{m.value}</div>
              <div className="text-storm-100 text-sm font-medium mt-1">{m.title}</div>
              <div className="text-storm-100/40 text-xs">{m.subtitle}</div>
            </GlassCard>
          )
        })}
      </div>

      <GlassCard className="p-5">
        <h3 className="text-storm-100 font-semibold mb-4">Police Station Efficiency</h3>
        {!stations?.available ? (
          <p className="text-storm-100/40 text-sm">Run precompute to generate station data.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-storm-100/50 text-xs uppercase">
                <tr>
                  <th className="text-left py-2">Station</th>
                  <th>Events</th>
                  <th>Score</th>
                  <th>Mean Hrs</th>
                  <th>Band</th>
                </tr>
              </thead>
              <tbody>
                {(stations.stations || []).slice(0, 15).map((s, i) => (
                  <tr key={i} className="border-t border-white/5 hover:bg-white/[0.03] transition-colors">
                    <td className="py-2 text-storm-100">{s['Police Station'] || s.police_station}</td>
                    <td className="text-center text-storm-100/70">{s['N Events'] ?? s.n_events}</td>
                    <td className="text-center text-storm-300 font-bold">{Number(s['Efficiency Score'] ?? s.efficiency_score ?? 0).toFixed(1)}</td>
                    <td className="text-center text-storm-100/70">{Number(s['Mean Actual Hrs'] ?? 0).toFixed(1)}</td>
                    <td className="text-center text-xs text-accent-success">{s['Performance Band']}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </GlassCard>

      <GlassCard className="p-5">
        <h3 className="text-storm-100 font-semibold mb-4">Cascade Chain Analysis</h3>
        {!cascades?.available ? (
          <p className="text-storm-100/40 text-sm">No cascade data available.</p>
        ) : (
          <>
            {cascades.top_insight && (
              <p
                className="text-sm rounded-lg p-3 mb-4 border"
                style={{
                  background: 'rgba(214, 154, 45, 0.1)',
                  borderColor: 'rgba(214, 154, 45, 0.2)',
                  color: ACCENT.gold,
                }}
              >
                {cascades.top_insight}
              </p>
            )}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-storm-100/50 text-xs uppercase">
                  <tr>
                    <th className="text-left py-2">Primary Cause</th>
                    <th>Cascade %</th>
                    <th>Triggered</th>
                    <th>Secondary</th>
                  </tr>
                </thead>
                <tbody>
                  {(cascades.rows || []).slice(0, 12).map((r) => (
                    <tr key={r.primary_cause} className="border-t border-white/5 hover:bg-white/[0.03] transition-colors">
                      <td className="py-2 text-storm-100">{r.primary_cause.replace('_', ' ')}</td>
                      <td className="text-center font-bold text-accent-amber">{r.cascade_pct}</td>
                      <td className="text-center text-storm-100/70">{r.triggered}</td>
                      <td className="text-storm-100/60 text-xs">{r.typical_secondary}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </GlassCard>
    </div>
  )
}