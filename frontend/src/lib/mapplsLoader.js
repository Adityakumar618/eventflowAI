const LOAD_TIMEOUT_MS = 14000
const POLL_INTERVAL_MS = 80

let loadPromise = null

function readStaticKeyFallback() {
  return import.meta.env.VITE_MAPPLS_STATIC_KEY || 'omkzebnezitmjsnjyfuhxgqpoflzwuzjrqgu'
}

function isSdkReady() {
  return Boolean(window.mappls?.Map)
}

function waitForSdk(timeoutMs = LOAD_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    if (isSdkReady()) {
      resolve(window.mappls)
      return
    }

    const started = Date.now()
    const timer = setInterval(() => {
      if (isSdkReady()) {
        clearInterval(timer)
        resolve(window.mappls)
        return
      }
      if (Date.now() - started >= timeoutMs) {
        clearInterval(timer)
        reject(new Error('Mappls SDK did not initialize'))
      }
    }, POLL_INTERVAL_MS)
  })
}

function injectScript(src, { callbackName } = {}) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-mappls-src="${src}"]`)
    if (existing) {
      waitForSdk(4000).then(resolve).catch(reject)
      return
    }

    const timeout = setTimeout(() => {
      reject(new Error(`Timed out loading Mappls SDK from ${src}`))
    }, LOAD_TIMEOUT_MS)

    const finishOk = () => {
      clearTimeout(timeout)
      waitForSdk(6000).then(resolve).catch(reject)
    }

    const finishErr = (err) => {
      clearTimeout(timeout)
      reject(err || new Error(`Failed to load Mappls SDK from ${src}`))
    }

    if (callbackName) {
      window[callbackName] = () => finishOk()
    }

    const script = document.createElement('script')
    script.setAttribute('data-mappls-src', src)
    script.async = true
    script.defer = true
    script.src = callbackName ? `${src}${src.includes('?') ? '&' : '?'}callback=${callbackName}` : src
    script.onload = () => {
      if (!callbackName) finishOk()
    }
    script.onerror = () => finishErr()
    document.head.appendChild(script)
  })
}

async function fetchMapConfig() {
  const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000'
  const res = await fetch(`${apiBase}/mappls/map-config`)
  if (!res.ok) throw new Error('Map config unavailable')
  return res.json()
}

export async function loadMapplsSdk() {
  if (isSdkReady()) return window.mappls
  if (loadPromise) return loadPromise

  loadPromise = (async () => {
    let config = null
    try {
      config = await fetchMapConfig()
    } catch {
      config = {
        static_key: readStaticKeyFallback(),
        rest_key: '99fcdc1f089f4dfaf2470df871d8c741',
        access_token: null,
      }
    }

    const staticKey = config.static_key || readStaticKeyFallback()
    const restKey = config.rest_key || '99fcdc1f089f4dfaf2470df871d8c741'
    const oauthToken = config.access_token

    const attempts = [
      `https://sdk.mappls.com/map/sdk/web?v=3.0&access_token=${encodeURIComponent(staticKey)}`,
      oauthToken
        ? `https://apis.mappls.com/advancedmaps/api/${encodeURIComponent(oauthToken)}/map_sdk?v=3.0&layer=vector`
        : null,
      `https://apis.mappls.com/advancedmaps/v1/${encodeURIComponent(restKey)}/map_sdk?layer=vector&v=3.0`,
    ].filter(Boolean)

    let lastError = null
    for (let i = 0; i < attempts.length; i += 1) {
      const src = attempts[i]
      const useCallback = src.includes('/map_sdk?')
      const callbackName = useCallback ? `mapplsReady_${i}` : null
      try {
        await injectScript(src, { callbackName })
        if (isSdkReady()) return window.mappls
      } catch (err) {
        lastError = err
      }
    }

    throw lastError || new Error('Unable to load Mappls SDK')
  })()

  try {
    return await loadPromise
  } catch (err) {
    loadPromise = null
    throw err
  }
}

export function resetMapplsLoader() {
  loadPromise = null
}