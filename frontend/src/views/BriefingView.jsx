import { useEffect, useState } from 'react'
import { Loader2, MapPin, Sunrise } from 'lucide-react'
import GlassCard from '../components/GlassCard'
import { api } from '../api'
import { riskColor } from '../theme/palette'

export default function BriefingView() {
  const [hour, setHour] = useState(8)
  const [briefing, setBriefing] = useState(null)
  const [trends, setTrends] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    Promise.all([api.riskBriefing(hour), api.trends()])
      .then(([b, t]) => {
        if (!cancelled) {
          setBriefing(b)
          setTrends(t)
        }
      })
      .catch(console.error)
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [hour])

  const monthly = trends?.available ? trends.monthly_summary || {} : {}
  const months = Object.keys(monthly)

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-storm-100 flex items-center gap-2">
            <Sunrise className="text-storm-300" size={24} /> Morning Briefing
          </h2>
          <p className="text-storm-100/50 text-sm mt-1">Risk pre-positioning before shift handover.</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-storm-100/60">Forecast hour: {hour}:00</span>
          <input type="range" min={0} max={23} value={hour} onChange={(e) => setHour(+e.target.value)} className="w-40 accent-storm-300" />
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-20"><Loader2 className="animate-spin text-storm-300" size={32} /></div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <GlassCard className="p-5 lg:col-span-2 min-h-[320px]">
              <h3 className="text-storm-100 font-semibold mb-4 flex items-center gap-2">
                <MapPin size={18} className="text-storm-300" /> Event Hotspot Map
              </h3>
              {!briefing?.hotspots?.length ? (
                <p className="text-storm-100/40 text-sm">{briefing?.message || 'No high-risk clusters for this hour.'}</p>
              ) : (
                <div className="relative map-canvas rounded-xl border h-64 overflow-hidden" style={{ borderColor: 'var(--border-subtle)' }}>
                  {briefing.hotspots.map((h, i) => (
                    <div
                      key={i}
                      title={`${h.corridor} · ${(h.risk * 100).toFixed(0)}%`}
                      className="absolute rounded-full border border-white/20"
                      style={{
                        left: `${((h.lon - 77.45) / 0.35) * 100}%`,
                        top: `${((13.15 - h.lat) / 0.25) * 100}%`,
                        width: `${8 + h.n_events * 0.5}px`,
                        height: `${8 + h.n_events * 0.5}px`,
                        backgroundColor: `rgba(224, 122, 95, ${0.35 + h.risk * 0.65})`,
                        transform: 'translate(-50%, -50%)',
                        boxShadow: `0 0 12px rgba(214, 154, 45, ${h.risk * 0.45})`,
                      }}
                    />
                  ))}
                  <div className="absolute bottom-2 right-2 text-xs text-storm-100/40">Bengaluru · {briefing.hotspots.length} clusters</div>
                </div>
              )}
            </GlassCard>

            <GlassCard className="p-5">
              <h3 className="text-storm-100 font-semibold mb-4">Top Risk Zones</h3>
              <div className="space-y-2 max-h-72 overflow-y-auto">
                {(briefing?.top_risks || []).slice(0, 8).map((r, i) => {
                  const pct = Math.round((r.risk_score || 0) * 100)
                  const color = riskColor(pct)
                  return (
                    <div key={i} className="bento-card rounded-xl p-3 border-l-4 hover:bg-white/[0.06] transition-colors" style={{ borderLeftColor: color }}>
                      <div className="flex justify-between items-center">
                        <span className="text-storm-100 text-sm font-medium truncate">#{i + 1} {(r.corridor || '').slice(0, 32)}</span>
                        <span className="font-bold text-sm" style={{ color }}>{pct}%</span>
                      </div>
                      <span className="text-storm-100/50 text-xs">{(r.top_cause || '').replace('_', ' ')} · {r.n_events || 0} events</span>
                    </div>
                  )
                })}
              </div>
            </GlassCard>
          </div>

          {trends?.available && (
            <GlassCard className="p-5">
              <h3 className="text-storm-100 font-semibold mb-4">Trend Intelligence</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <p className="text-xs text-storm-100/50 uppercase mb-2">Event Frequency · {trends.frequency_trend?.direction}</p>
                  <div className="flex items-end gap-1 h-24">
                    {months.map((m) => {
                      const max = Math.max(...months.map((x) => monthly[x].count), 1)
                      const h = (monthly[m].count / max) * 100
                      return <div key={m} title={`${m}: ${monthly[m].count}`} className="flex-1 bg-storm-300/80 rounded-t hover:bg-storm-300 transition-colors" style={{ height: `${h}%` }} />
                    })}
                  </div>
                  <p className="text-xs text-storm-100/40 mt-2">{trends.frequency_trend?.insight}</p>
                </div>
                <div>
                  <p className="text-xs text-storm-100/50 uppercase mb-2">Median Duration · {trends.duration_trend?.direction}</p>
                  <div className="flex items-end gap-1 h-24">
                    {months.map((m) => {
                      const vals = months.map((x) => monthly[x].median_hrs)
                      const max = Math.max(...vals, 1)
                      const h = (monthly[m].median_hrs / max) * 100
                      return <div key={m} className="flex-1 bg-accent-amber/80 rounded-t hover:bg-accent-amber transition-colors" style={{ height: `${h}%` }} />
                    })}
                  </div>
                  <p className="text-xs text-storm-100/40 mt-2">{trends.duration_trend?.insight}</p>
                </div>
                <div>
                  <p className="text-xs text-storm-100/50 uppercase mb-2">Cause Growth</p>
                  <div className="space-y-1 text-sm max-h-28 overflow-y-auto">
                    {Object.entries(trends.cause_trends || {})
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 6)
                      .map(([cause, slope]) => (
                        <div key={cause} className="flex justify-between">
                          <span className="text-storm-100/70">{cause.replace('_', ' ')}</span>
                          <span className={slope > 0 ? 'text-accent-terracotta' : 'text-accent-success'}>
                            {slope > 0 ? '+' : ''}{slope.toFixed(2)}
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            </GlassCard>
          )}
        </>
      )}
    </div>
  )
}