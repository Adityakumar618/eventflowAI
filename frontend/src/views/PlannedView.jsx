import { useEffect, useState } from 'react'
import { Calendar, Loader2 } from 'lucide-react'
import GlassCard from '../components/GlassCard'
import FormSelect from '../components/FormSelect'
import { api } from '../api'
import { ACCENT, cascadeColor, stisColor } from '../theme/palette'

export default function PlannedView() {
  const [summary, setSummary] = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [dossier, setDossier] = useState(null)
  const [filter, setFilter] = useState('All')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.plannedSummary(), api.plannedAnalytics()])
      .then(([s, a]) => {
        setSummary(s)
        setAnalytics(a)
        if (s.events?.length) setSelectedId(s.events[0].id)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (selectedId == null) return
    api.plannedDossier(selectedId).then(setDossier).catch(console.error)
  }, [selectedId])

  const events = (summary?.events || []).filter(
    (e) => filter === 'All' || e.event_cause === filter,
  )

  if (loading) {
    return <div className="flex justify-center py-20"><Loader2 className="animate-spin text-storm-300" size={32} /></div>
  }

  if (!summary?.available) {
    return (
      <GlassCard className="p-8 text-center text-storm-100/40">
        Run precompute pipeline to generate planned event data.
      </GlassCard>
    )
  }

  const s = summary.summary

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-storm-100 flex items-center gap-2">
          <Calendar className="text-storm-300" size={24} /> Planned Events
        </h2>
        <p className="text-storm-100/50 text-sm mt-1">24-hour pre-event dossiers for processions & VIP movements.</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          [s.total_planned, 'Total Planned', ACCENT.gold],
          [s.need_closure, 'Need Closure', ACCENT.terracotta],
          [`${s.median_duration_hrs?.toFixed(1) || '—'}h`, 'Median Duration', ACCENT.amber],
          [s.night_events, 'Night Events', ACCENT.bronze],
        ].map(([val, label, color]) => (
          <GlassCard key={label} className="p-4 text-center bento-card !bg-white/[0.04]">
            <div className="text-2xl font-black" style={{ color }}>{val}</div>
            <div className="text-storm-100/50 text-xs mt-1">{label}</div>
          </GlassCard>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <GlassCard className="p-5 space-y-3">
          <h3 className="text-storm-100 font-semibold">Select Event</h3>
          <FormSelect
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="text-sm"
          >
            <option>All</option>
            {(summary.types || []).map((t) => <option key={t} value={t}>{t.replace('_', ' ')}</option>)}
          </FormSelect>
          <div className="max-h-64 overflow-y-auto space-y-1">
            {events.map((e) => (
              <button
                key={e.id}
                onClick={() => setSelectedId(e.id)}
                className={`w-full text-left text-xs p-2 rounded-lg transition-colors ${
                  selectedId === e.id
                    ? 'bg-storm-300/20 border border-storm-300/40 text-storm-100'
                    : 'bg-white/[0.03] border border-white/[0.06] text-storm-100/70 hover:bg-white/[0.06]'
                }`}
              >
                {e.label}
              </button>
            ))}
          </div>
        </GlassCard>

        <GlassCard className="p-5 lg:col-span-2">
          {dossier?.available ? (
            <>
              <h3 className="text-storm-100 font-bold text-lg mb-1">24-Hour Pre-Event Dossier</h3>
              <p className="text-storm-100/50 text-sm mb-4">{dossier.cause.replace('_', ' ').toUpperCase()}</p>
              <div className="grid grid-cols-2 gap-3 mb-4">
                {[
                  ['Predicted Duration', `${dossier.median_hrs}h`, ACCENT.gold, `P10 ${dossier.p10_hrs}h · P90 ${dossier.p90_hrs}h`],
                  ['STIS', `${dossier.stis}/10`, stisColor(dossier.stis), dossier.stis_label],
                  ['Deployment', `${dossier.officers} officers`, ACCENT.success, dossier.needs_closure ? '+ Barricades' : 'Standard'],
                  ['Cascade Risk', `${(dossier.cascade_prob * 100).toFixed(0)}%`, cascadeColor(dossier.cascade_prob), dossier.secondary_causes?.join(', ') || 'Low'],
                ].map(([label, val, color, sub]) => (
                  <div key={label} className="bento-card rounded-xl p-4 hover:bg-white/[0.06] transition-colors">
                    <div className="text-storm-100/50 text-xs uppercase">{label}</div>
                    <div className="text-2xl font-black mt-1" style={{ color }}>{val}</div>
                    <div className="text-storm-100/40 text-xs mt-1">{sub}</div>
                  </div>
                ))}
              </div>
              <div className="bento-card border-l-4 border-storm-500 rounded-xl p-4">
                <div className="text-storm-100/50 text-xs uppercase mb-2">Executive Summary</div>
                <p className="text-storm-100 text-sm leading-relaxed">{dossier.executive_summary}</p>
              </div>
            </>
          ) : (
            <p className="text-storm-100/40">Select an event to view dossier.</p>
          )}
        </GlassCard>
      </div>

      {analytics?.available && (
        <GlassCard className="p-5">
          <h3 className="text-storm-100 font-semibold mb-4">Historical Intelligence</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <p className="text-xs text-storm-100/50 mb-2">Median Duration by Type</p>
              <div className="space-y-2">
                {(analytics.duration_by_cause || []).map((d) => (
                  <div key={d.event_cause} className="flex items-center gap-2">
                    <span className="text-xs text-storm-100/60 w-28 truncate">{d.event_cause}</span>
                    <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-storm-300 rounded-full" style={{ width: `${Math.min(d.duration_hrs * 10, 100)}%` }} />
                    </div>
                    <span className="text-xs text-storm-300 w-10">{d.duration_hrs?.toFixed(1)}h</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <p className="text-xs text-storm-100/50 mb-2">Events by Hour</p>
              <div className="flex items-end gap-0.5 h-24">
                {Array.from({ length: 24 }, (_, h) => {
                  const row = (analytics.events_by_hour || []).find((x) => x.hour === h)
                  const count = row?.count || 0
                  const max = Math.max(...(analytics.events_by_hour || []).map((x) => x.count), 1)
                  return (
                    <div
                      key={h}
                      title={`${h}:00 — ${count}`}
                      className="flex-1 bg-accent-gold/70 rounded-t min-h-[2px] hover:bg-accent-gold transition-colors"
                      style={{ height: `${(count / max) * 100}%` }}
                    />
                  )
                })}
              </div>
            </div>
          </div>
        </GlassCard>
      )}
    </div>
  )
}