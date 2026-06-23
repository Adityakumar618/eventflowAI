import { MapPin } from 'lucide-react'

const HOTSPOTS = [
  { name: 'Silk Board', lat: 12.9177, lon: 77.6228, risk: 'HIGH' },
  { name: 'Hebbal Flyover', lat: 13.0354, lon: 77.5910, risk: 'HIGH' },
  { name: 'Marathahalli', lat: 12.9592, lon: 77.6974, risk: 'MEDIUM' },
  { name: 'KR Puram', lat: 13.0053, lon: 77.6946, risk: 'MEDIUM' },
  { name: 'Mekhri Circle', lat: 13.0090, lon: 77.5770, risk: 'HIGH' },
  { name: 'Yeshwanthpur', lat: 13.0267, lon: 77.5361, risk: 'MEDIUM' },
]

const RISK_COLORS = { HIGH: '#E07A5F', MEDIUM: '#D69A2D', LOW: '#4CAF82' }

const LON_MIN = 77.45
const LON_MAX = 77.80
const LAT_MIN = 12.85
const LAT_MAX = 13.15

function toPercent(lat, lon) {
  return {
    left: `${((lon - LON_MIN) / (LON_MAX - LON_MIN)) * 100}%`,
    top: `${((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * 100}%`,
  }
}

function fromPercent(clientX, clientY, rect) {
  const x = (clientX - rect.left) / rect.width
  const y = (clientY - rect.top) / rect.height
  const lon = LON_MIN + x * (LON_MAX - LON_MIN)
  const lat = LAT_MAX - y * (LAT_MAX - LAT_MIN)
  return {
    lat: Math.round(lat * 1000) / 1000,
    lon: Math.round(lon * 1000) / 1000,
  }
}

export default function BengaluruMap({
  lat,
  lon,
  eventCause,
  onLocationChange,
  height = 280,
  label = 'Bengaluru Traffic Map',
  interactive = true,
}) {
  const eventPos = toPercent(lat, lon)

  const handleClick = (e) => {
    if (!interactive || !onLocationChange) return
    const rect = e.currentTarget.getBoundingClientRect()
    const { lat: newLat, lon: newLon } = fromPercent(e.clientX, e.clientY, rect)
    onLocationChange(newLat, newLon)
  }

  return (
    <div
      className="relative rounded-xl overflow-hidden border map-canvas"
      style={{ height, borderColor: 'var(--border-subtle)' }}
    >
      <div
        role={interactive ? 'button' : undefined}
        tabIndex={interactive ? 0 : undefined}
        onClick={handleClick}
        onKeyDown={(e) => {
          if (!interactive) return
          const step = 0.005
          if (e.key === 'ArrowUp') onLocationChange?.(lat + step, lon)
          if (e.key === 'ArrowDown') onLocationChange?.(lat - step, lon)
          if (e.key === 'ArrowLeft') onLocationChange?.(lat, lon - step)
          if (e.key === 'ArrowRight') onLocationChange?.(lat, lon + step)
        }}
        className={`relative h-full w-full ${interactive ? 'cursor-crosshair' : ''}`}
        title={interactive ? 'Click to set event location' : undefined}
      >
        {/* Grid lines */}
        <svg className="absolute inset-0 h-full w-full opacity-20 pointer-events-none" preserveAspectRatio="none">
          {Array.from({ length: 5 }, (_, i) => (
            <line key={`v${i}`} x1={`${(i + 1) * 20}%`} y1="0" x2={`${(i + 1) * 20}%`} y2="100%" stroke="var(--text-muted)" strokeWidth="0.5" />
          ))}
          {Array.from({ length: 4 }, (_, i) => (
            <line key={`h${i}`} x1="0" y1={`${(i + 1) * 25}%`} x2="100%" y2={`${(i + 1) * 25}%`} stroke="var(--text-muted)" strokeWidth="0.5" />
          ))}
        </svg>

        {/* Hotspot markers */}
        {HOTSPOTS.map((h) => {
          const pos = toPercent(h.lat, h.lon)
          const color = RISK_COLORS[h.risk]
          return (
            <div
              key={h.name}
              title={`${h.name} · ${h.risk} RISK`}
              className="absolute rounded-full border-2 pointer-events-none"
              style={{
                ...pos,
                width: 14,
                height: 14,
                transform: 'translate(-50%, -50%)',
                backgroundColor: `${color}55`,
                borderColor: color,
                boxShadow: `0 0 10px ${color}66`,
              }}
            />
          )
        })}

        {/* Event pin */}
        <div
          className="absolute z-10 pointer-events-none flex flex-col items-center"
          style={{ ...eventPos, transform: 'translate(-50%, -100%)' }}
        >
          <div
            className="flex h-8 w-8 items-center justify-center rounded-full shadow-lg"
            style={{ background: 'var(--accent)', color: 'var(--accent-ink)' }}
          >
            <MapPin size={16} />
          </div>
          <div
            className="mt-1 rounded-md px-2 py-0.5 text-[10px] font-medium whitespace-nowrap backdrop-blur-sm"
            style={{ background: 'var(--map-badge-bg)', color: 'var(--text-primary)' }}
          >
            {(eventCause || 'event').replace(/_/g, ' ')}
          </div>
        </div>

        <div
          className="absolute bottom-2 right-2 text-[10px] pointer-events-none"
          style={{ color: 'var(--text-faint)' }}
        >
          {lat.toFixed(3)}, {lon.toFixed(3)}
        </div>
      </div>

      <div
        className="absolute top-2 left-2 rounded-lg px-2 py-1 text-xs flex items-center gap-1 backdrop-blur-sm"
        style={{ background: 'var(--map-badge-bg)', color: 'var(--text-muted)' }}
      >
        <MapPin size={10} /> {label}
      </div>

      {interactive && (
        <div
          className="absolute bottom-2 left-2 text-[10px] pointer-events-none"
          style={{ color: 'var(--text-faint)' }}
        >
          Click map to set coordinates
        </div>
      )}
    </div>
  )
}