import { useEffect, useId, useRef, useState } from 'react'
import { AlertTriangle, Layers, Loader2, MapPin } from 'lucide-react'
import { loadMapplsSdk } from '../lib/mapplsLoader'
import { setMapTrafficClosures, setMapTrafficOverlay, TRAFFIC_ROAD_LEGEND } from '../lib/mapplsTraffic'
import BengaluruMap from './BengaluruMap'
import CorridorTrafficPanel from './CorridorTrafficPanel'

const HOTSPOTS = [
  { name: 'Silk Board', lat: 12.9177, lon: 77.6228, risk: 'HIGH' },
  { name: 'Hebbal Flyover', lat: 13.0354, lon: 77.5910, risk: 'HIGH' },
  { name: 'Marathahalli', lat: 12.9592, lon: 77.6974, risk: 'MEDIUM' },
  { name: 'KR Puram', lat: 13.0053, lon: 77.6946, risk: 'MEDIUM' },
  { name: 'Mekhri Circle', lat: 13.0090, lon: 77.5770, risk: 'HIGH' },
  { name: 'Yeshwanthpur', lat: 13.0267, lon: 77.5361, risk: 'MEDIUM' },
]

const RISK_COLORS = { HIGH: '#E07A5F', MEDIUM: '#D69A2D', LOW: '#4CAF82' }

function extractLatLng(event) {
  if (!event) return null
  if (event.latLng) return { lat: event.latLng.lat, lon: event.latLng.lng }
  if (event.lngLat) return { lat: event.lngLat.lat, lon: event.lngLat.lng }
  if (typeof event.lat === 'number' && typeof event.lng === 'number') {
    return { lat: event.lat, lon: event.lng }
  }
  return null
}

export default function MapplsEventMap({
  lat,
  lon,
  eventCause,
  onLocationChange,
  height = 320,
  showTrafficPanel = true,
}) {
  const reactId = useId().replace(/:/g, '')
  const mapContainerId = `mappls-${reactId}`
  const mapRef = useRef(null)
  const mapInstanceRef = useRef(null)
  const markerRef = useRef(null)
  const hotspotMarkersRef = useRef([])
  const onLocationChangeRef = useRef(onLocationChange)
  const mapReadyRef = useRef(false)

  const [status, setStatus] = useState('loading')
  const [errorMsg, setErrorMsg] = useState('')
  const [trafficEnabled, setTrafficEnabled] = useState(true)
  const [trafficSupported, setTrafficSupported] = useState(false)

  useEffect(() => {
    onLocationChangeRef.current = onLocationChange
  }, [onLocationChange])

  useEffect(() => {
    let cancelled = false
    let resizeObserver = null

    const init = async () => {
      setStatus('loading')
      setErrorMsg('')
      try {
        const mappls = await loadMapplsSdk()
        if (cancelled || !mapRef.current) return

        if (mapInstanceRef.current) {
          try { mapInstanceRef.current.remove?.() } catch (_) {}
          mapInstanceRef.current = null
        }

        const map = new mappls.Map(mapContainerId, {
          center: { lat: 12.9716, lng: 77.5946 },
          zoom: 11,
        })
        mapInstanceRef.current = map

        const placeHotspots = () => {
          hotspotMarkersRef.current.forEach((m) => {
            try { m.remove?.() } catch (_) {}
          })
          hotspotMarkersRef.current = []

          HOTSPOTS.forEach((h) => {
            try {
              const marker = new mappls.Marker({
                map,
                position: { lat: h.lat, lng: h.lon },
                popupHtml: `<div style="font-family:Inter,sans-serif;padding:8px 10px"><b>${h.name}</b><br><span style="color:${RISK_COLORS[h.risk]}">${h.risk} RISK</span></div>`,
                width: 22,
                height: 22,
                offset: [0, -8],
              })
              hotspotMarkersRef.current.push(marker)
            } catch (_) {}
          })
        }

        const upsertEventMarker = (nextLat, nextLon) => {
          try {
            if (markerRef.current) markerRef.current.remove?.()
            markerRef.current = new mappls.Marker({
              map,
              position: { lat: nextLat, lng: nextLon },
              draggable: Boolean(onLocationChangeRef.current),
              popupHtml: `<div style="font-family:Inter,sans-serif;padding:8px 10px"><b>New Event</b><br>${(eventCause || 'event').replace(/_/g, ' ')}<br><small>${nextLat.toFixed(4)}, ${nextLon.toFixed(4)}</small></div>`,
              width: 34,
              height: 34,
              offset: [0, -14],
            })

            if (onLocationChangeRef.current) {
              markerRef.current.addListener?.('dragend', () => {
                const pos = markerRef.current?.getPosition?.()
                if (!pos) return
                const next = extractLatLng(pos) || { lat: pos.lat, lon: pos.lng }
                onLocationChangeRef.current?.(next.lat, next.lon)
              })
            }
          } catch (_) {}
        }

        const ready = () => {
          if (cancelled || mapReadyRef.current) return
          mapReadyRef.current = true
          placeHotspots()
          upsertEventMarker(lat, lon)
          map.setCenter?.({ lat, lng: lon })
          const overlayOk = setMapTrafficOverlay(map, true)
          setMapTrafficClosures(map, true)
          setTrafficSupported(overlayOk)
          setTrafficEnabled(overlayOk)
          setStatus('ready')
        }

        if (map.addListener) {
          map.addListener('load', ready)
          map.addListener('error', () => {
            if (!cancelled) {
              setStatus('error')
              setErrorMsg('Mappls map failed to render tiles')
            }
          })
        }

        if (map.loaded?.()) {
          ready()
        } else {
          setTimeout(() => {
            if (!cancelled && !mapReadyRef.current) ready()
          }, 2500)
        }

        map.addListener?.('click', (evt) => {
          if (!onLocationChangeRef.current) return
          const next = extractLatLng(evt)
          if (!next) return
          onLocationChangeRef.current(next.lat, next.lon)
          upsertEventMarker(next.lat, next.lon)
        })

        if (window.ResizeObserver && mapRef.current) {
          resizeObserver = new ResizeObserver(() => {
            try { map.resize?.() } catch (_) {}
          })
          resizeObserver.observe(mapRef.current)
        }
      } catch (err) {
        if (!cancelled) {
          setStatus('error')
          setErrorMsg(err?.message || 'Mappls SDK unavailable')
        }
      }
    }

    init()

    return () => {
      cancelled = true
      mapReadyRef.current = false
      resizeObserver?.disconnect()
      hotspotMarkersRef.current.forEach((m) => {
        try { m.remove?.() } catch (_) {}
      })
      hotspotMarkersRef.current = []
      try { markerRef.current?.remove?.() } catch (_) {}
      markerRef.current = null
      try { mapInstanceRef.current?.remove?.() } catch (_) {}
      mapInstanceRef.current = null
    }
  }, [mapContainerId])

  useEffect(() => {
    const map = mapInstanceRef.current
    if (!map || status !== 'ready') return

    try {
      if (markerRef.current) markerRef.current.remove?.()
      const mappls = window.mappls
      markerRef.current = new mappls.Marker({
        map,
        position: { lat, lng: lon },
        draggable: Boolean(onLocationChange),
        popupHtml: `<div style="font-family:Inter,sans-serif;padding:8px 10px"><b>New Event</b><br>${(eventCause || 'event').replace(/_/g, ' ')}<br><small>${lat.toFixed(4)}, ${lon.toFixed(4)}</small></div>`,
        width: 34,
        height: 34,
        offset: [0, -14],
      })
      if (onLocationChange) {
        markerRef.current.addListener?.('dragend', () => {
          const pos = markerRef.current?.getPosition?.()
          if (!pos) return
          const next = extractLatLng(pos) || { lat: pos.lat, lon: pos.lng }
          onLocationChange(next.lat, next.lon)
        })
      }
      map.setCenter?.({ lat, lng: lon })
    } catch (_) {}
  }, [lat, lon, eventCause, onLocationChange, status])

  useEffect(() => {
    const map = mapInstanceRef.current
    if (!map || status !== 'ready' || !trafficSupported) return
    setMapTrafficOverlay(map, trafficEnabled)
    if (trafficEnabled) setMapTrafficClosures(map, true)
  }, [trafficEnabled, status, trafficSupported])

  const toggleTraffic = () => {
    const next = !trafficEnabled
    setTrafficEnabled(next)
    if (mapInstanceRef.current) {
      const ok = setMapTrafficOverlay(mapInstanceRef.current, next)
      if (!ok) setTrafficSupported(false)
    }
  }

  const trafficPanel = showTrafficPanel ? <CorridorTrafficPanel lat={lat} lon={lon} /> : null

  if (status === 'error') {
    return (
      <div className="space-y-3">
        <div
          className="rounded-lg border px-3 py-2 text-xs flex items-start gap-2"
          style={{ borderColor: 'var(--border-subtle)', color: 'var(--text-muted)', background: 'var(--bento-bg)' }}
        >
          <AlertTriangle size={14} className="shrink-0 mt-0.5 text-accent-amber" />
          <span>Live Mappls map unavailable ({errorMsg}). Showing interactive fallback — click to set coordinates. Corridor delays still load from Mappls Route ADV.</span>
        </div>
        <BengaluruMap
          lat={lat}
          lon={lon}
          eventCause={eventCause}
          onLocationChange={onLocationChange}
          height={height}
          label="Bengaluru Fallback Map"
        />
        {trafficPanel}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div
        className="relative rounded-xl overflow-hidden border"
        style={{ height, borderColor: 'var(--border-subtle)' }}
      >
        <div
          id={mapContainerId}
          ref={mapRef}
          className="h-full w-full"
          style={{ minHeight: height, background: '#0a0f1a' }}
        />

        {status === 'loading' && (
          <div
            className="absolute inset-0 flex items-center justify-center backdrop-blur-sm"
            style={{ background: 'rgba(0,0,0,0.45)' }}
          >
            <div className="flex flex-col items-center gap-2 text-storm-100/60">
              <Loader2 size={24} className="animate-spin" />
              <span className="text-xs">Loading Mappls Bengaluru map…</span>
            </div>
          </div>
        )}

        <div
          className="pointer-events-none absolute top-2 left-2 rounded-lg px-2 py-1 text-xs flex items-center gap-1 backdrop-blur-sm"
          style={{ background: 'var(--map-badge-bg)', color: 'var(--text-muted)' }}
        >
          <MapPin size={10} /> Mappls · Bengaluru Live
        </div>

        {status === 'ready' && trafficSupported && (
          <button
            type="button"
            onClick={toggleTraffic}
            className="absolute top-2 right-2 z-10 flex items-center gap-1 rounded-lg px-2 py-1 text-[10px] font-medium backdrop-blur-sm transition-opacity hover:opacity-90"
            style={{
              background: trafficEnabled ? 'rgba(76, 175, 130, 0.25)' : 'var(--map-badge-bg)',
              color: trafficEnabled ? '#4CAF82' : 'var(--text-muted)',
              border: `1px solid ${trafficEnabled ? 'rgba(76, 175, 130, 0.45)' : 'var(--border-subtle)'}`,
            }}
            title="Toggle live traffic road colours"
          >
            <Layers size={11} />
            Traffic {trafficEnabled ? 'ON' : 'OFF'}
          </button>
        )}

        {status === 'ready' && trafficEnabled && trafficSupported && (
          <div
            className="pointer-events-none absolute bottom-2 right-2 rounded-lg px-2 py-1.5 backdrop-blur-sm"
            style={{ background: 'var(--map-badge-bg)' }}
          >
            <div className="text-[9px] uppercase tracking-wide mb-1" style={{ color: 'var(--text-faint)' }}>
              Road traffic
            </div>
            <div className="flex items-center gap-2">
              {TRAFFIC_ROAD_LEGEND.map((item) => (
                <span key={item.label} className="flex items-center gap-1 text-[9px]" style={{ color: 'var(--text-muted)' }}>
                  <span className="inline-block w-2.5 h-1 rounded-sm" style={{ background: item.color }} />
                  {item.label}
                </span>
              ))}
            </div>
          </div>
        )}

        {onLocationChange && status === 'ready' && (
          <div
            className="pointer-events-none absolute bottom-2 left-2 text-[10px]"
            style={{ color: 'var(--text-faint)' }}
          >
            Click map or drag pin to set coordinates
          </div>
        )}
      </div>

      {trafficPanel}
    </div>
  )
}