const API_BASE = import.meta.env.VITE_API_URL || 'https://eventflowai-1uav.onrender.com'

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export const api = {
  health: () => request('/health'),
  options: () => request('/meta/options'),
  predictEvent: (body) => request('/predict/event', { method: 'POST', body: JSON.stringify(body) }),
  triage: (body) => request('/triage/optimize', { method: 'POST', body: JSON.stringify(body) }),
  prescriptive: (body) => request('/prescriptive/recommend', { method: 'POST', body: JSON.stringify(body) }),
  riskBriefing: (hour) => request(`/briefing/risk?hour=${hour}`),
  trends: () => request('/briefing/trends'),
  stations: () => request('/audit/stations'),
  metrics: () => request('/audit/metrics'),
  cascades: () => request('/audit/cascades'),
  plannedSummary: () => request('/planned/summary'),
  plannedDossier: (id) => request(`/planned/dossier/${id}`),
  plannedAnalytics: () => request('/planned/analytics'),
  mapplsConfig: () => request('/mappls/map-config'),
  trafficSnapshot: (lat, lon) => {
    const params = new URLSearchParams()
    if (lat != null) params.set('lat', String(lat))
    if (lon != null) params.set('lon', String(lon))
    const q = params.toString()
    return request(`/mappls/traffic-snapshot${q ? `?${q}` : ''}`)
  },
}