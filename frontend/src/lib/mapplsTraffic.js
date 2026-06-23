/** Mappls live traffic overlay helpers (defensive — SDK method names vary by version). */

export const TRAFFIC_LEVEL_COLORS = {
  LOW: '#4CAF82',
  MEDIUM: '#D69A2D',
  HIGH: '#E07A5F',
  UNKNOWN: '#8F8A82',
}

export const TRAFFIC_ROAD_LEGEND = [
  { color: '#4CAF82', label: 'Free flow' },
  { color: '#D69A2D', label: 'Moderate' },
  { color: '#E07A5F', label: 'Heavy' },
]

function tryCall(map, fnNames, ...args) {
  for (const name of fnNames) {
    const fn = map?.[name]
    if (typeof fn === 'function') {
      fn.apply(map, args)
      return true
    }
  }
  return false
}

/** Toggle live road-colour traffic overlay on a Mappls map instance. */
export function setMapTrafficOverlay(map, enabled) {
  if (!map) return false
  try {
    if (tryCall(map, ['enableTraffic', 'showTraffic', 'setTraffic'], enabled)) {
      return true
    }
    if (enabled && typeof map.addTraffic === 'function') {
      map.addTraffic()
      return true
    }
    if (!enabled && typeof map.removeTraffic === 'function') {
      map.removeTraffic()
      return true
    }
  } catch (_) {}
  return false
}

/** Optional road-closure layer (when supported by SDK). */
export function setMapTrafficClosures(map, enabled) {
  if (!map) return false
  try {
    return tryCall(map, ['enableTrafficClosure', 'showTrafficClosure', 'setTrafficClosure'], enabled)
  } catch (_) {}
  return false
}