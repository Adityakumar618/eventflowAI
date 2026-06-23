import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertTriangle, Clock, Loader2, RefreshCw, Route } from 'lucide-react'
import { api } from '../api'
import { TRAFFIC_LEVEL_COLORS } from '../lib/mapplsTraffic'

function formatDelay(mins) {
  if (mins == null) return '—'
  if (mins < 1) return '<1 min'
  return `+${Math.round(mins)} min`
}

function formatUpdatedAt(epochSec) {
  if (!epochSec) return ''
  const diff = Math.max(0, Math.floor(Date.now() / 1000) - epochSec)
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return new Date(epochSec * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function LevelChip({ level }) {
  const color = TRAFFIC_LEVEL_COLORS[level] || TRAFFIC_LEVEL_COLORS.UNKNOWN
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
      style={{ background: `${color}22`, color, border: `1px solid ${color}44` }}
    >
      {level || 'UNKNOWN'}
    </span>
  )
}

function CorridorRow({ corridor }) {
  const level = corridor.congestion_level || 'UNKNOWN'
  const color = TRAFFIC_LEVEL_COLORS[level] || TRAFFIC_LEVEL_COLORS.UNKNOWN
  const live = corridor.source === 'mappls_live'

  return (
    <div
      className="flex items-center justify-between gap-2 rounded-lg px-2.5 py-2 text-xs"
      style={{ background: 'var(--bento-bg)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="min-w-0 flex-1">
        <div className="truncate font-medium" style={{ color: 'var(--text-primary)' }}>
          {corridor.name}
        </div>
        <div className="mt-0.5 flex items-center gap-1.5" style={{ color: 'var(--text-faint)' }}>
          <LevelChip level={level} />
          {corridor.available && corridor.distance_km != null && (
            <span>{corridor.distance_km} km</span>
          )}
          {corridor.available && (
            <span className="text-[10px]">{live ? 'Mappls live' : 'Route + ISEC est.'}</span>
          )}
        </div>
      </div>
      <div className="shrink-0 text-right">
        <div className="font-bold tabular-nums" style={{ color }}>
          {corridor.available ? formatDelay(corridor.delay_mins) : 'N/A'}
        </div>
        {corridor.available && corridor.traffic_mins != null && (
          <div className="text-[10px]" style={{ color: 'var(--text-faint)' }}>
            {corridor.traffic_mins} min total
          </div>
        )}
      </div>
    </div>
  )
}

export default function CorridorTrafficPanel({ lat, lon }) {
  const [snapshot, setSnapshot] = useState(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const pollRef = useRef(null)

  const fetchSnapshot = useCallback(async (silent = false) => {
    if (!silent) setStatus('loading')
    try {
      const data = await api.trafficSnapshot(lat, lon)
      setSnapshot(data)
      setError('')
      setStatus('ready')
      return data
    } catch (err) {
      setError(err?.message || 'Traffic data unavailable')
      setStatus('error')
      return null
    }
  }, [lat, lon])

  useEffect(() => {
    let cancelled = false

    const run = async () => {
      const data = await fetchSnapshot(false)
      if (cancelled || !data) return

      const intervalMs = (data.refresh_sec || 120) * 1000
      pollRef.current = window.setInterval(() => {
        fetchSnapshot(true)
      }, intervalMs)
    }

    run()

    return () => {
      cancelled = true
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [fetchSnapshot])

  const summary = snapshot?.summary
  const corridors = snapshot?.corridors || []
  const eventProbe = snapshot?.event_probe

  return (
    <div
      className="rounded-xl border p-3 space-y-3"
      style={{ borderColor: 'var(--border-subtle)', background: 'var(--surface-card)' }}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <h4
            className="text-xs font-semibold flex items-center gap-1.5"
            style={{ color: 'var(--text-primary)' }}
          >
            <Route size={13} className="text-accent-amber" />
            Bengaluru corridor delays
          </h4>
          <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-faint)' }}>
            Live Route ADV probes · refreshes every {snapshot?.refresh_sec || 120}s
          </p>
        </div>
        <button
          type="button"
          onClick={() => fetchSnapshot(false)}
          disabled={status === 'loading'}
          className="shrink-0 rounded-lg p-1.5 transition-colors hover:opacity-80 disabled:opacity-40"
          style={{ background: 'var(--bento-bg)', color: 'var(--text-muted)' }}
          title="Refresh traffic snapshot"
        >
          {status === 'loading' ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <RefreshCw size={14} />
          )}
        </button>
      </div>

      {status === 'loading' && !snapshot && (
        <div className="flex items-center gap-2 text-xs py-4 justify-center" style={{ color: 'var(--text-muted)' }}>
          <Loader2 size={14} className="animate-spin" />
          Probing Mappls corridors…
        </div>
      )}

      {error && !snapshot && (
        <div
          className="flex items-start gap-2 rounded-lg px-2.5 py-2 text-xs"
          style={{ background: 'var(--bento-bg)', color: 'var(--text-muted)' }}
        >
          <AlertTriangle size={14} className="shrink-0 mt-0.5 text-accent-amber" />
          {error}
        </div>
      )}

      {snapshot && (
        <>
          <div
            className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg px-2.5 py-2 text-[11px]"
            style={{ background: 'var(--bento-bg)', color: 'var(--text-muted)' }}
          >
            <span className="flex items-center gap-1">
              <Clock size={11} />
              Updated {formatUpdatedAt(snapshot.updated_at)}
            </span>
            {summary?.corridors_live != null && (
              <span>{summary.corridors_live}/{corridors.length} corridors live</span>
            )}
            {summary?.worst_corridor && (
              <span>
                Worst: <strong style={{ color: TRAFFIC_LEVEL_COLORS[summary.worst_level] }}>{summary.worst_corridor}</strong>
                {summary.worst_delay_mins != null && ` (${formatDelay(summary.worst_delay_mins)})`}
              </span>
            )}
          </div>

          {eventProbe?.available && (
            <div
              className="rounded-lg px-2.5 py-2 text-xs flex items-center justify-between"
              style={{
                background: `${TRAFFIC_LEVEL_COLORS[eventProbe.congestion_level] || TRAFFIC_LEVEL_COLORS.MEDIUM}14`,
                border: `1px solid ${TRAFFIC_LEVEL_COLORS[eventProbe.congestion_level] || TRAFFIC_LEVEL_COLORS.MEDIUM}33`,
              }}
            >
              <span style={{ color: 'var(--text-primary)' }}>
                {eventProbe.label || 'Event location probe'}
              </span>
              <div className="flex items-center gap-2">
                <LevelChip level={eventProbe.congestion_level} />
                <span className="font-bold" style={{ color: TRAFFIC_LEVEL_COLORS[eventProbe.congestion_level] }}>
                  {formatDelay(eventProbe.delay_mins)}
                </span>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {corridors.map((c) => (
              <CorridorRow key={c.id} corridor={c} />
            ))}
          </div>

          {error && (
            <p className="text-[10px]" style={{ color: 'var(--text-faint)' }}>
              Last refresh failed — showing cached data. {error}
            </p>
          )}
        </>
      )}
    </div>
  )
}