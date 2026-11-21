const STORAGE_KEY = 'sentineldesk_api_base'

export function isStaticHost() {
  if (typeof window === 'undefined') return false
  return window.location.hostname.endsWith('github.io')
}

export function getApiBase() {
  if (typeof window !== 'undefined') {
    const params = new URLSearchParams(window.location.search)
    const fromQuery = params.get('api')
    if (fromQuery) {
      const cleaned = fromQuery.trim().replace(/\/+$/, '')
      try {
        window.localStorage.setItem(STORAGE_KEY, cleaned)
      } catch {}
      return cleaned
    }
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY)
      if (saved) return saved.trim().replace(/\/+$/, '')
    } catch {}
  }

  const envBase = (import.meta.env.VITE_API_BASE || '').trim().replace(/\/+$/, '')
  if (envBase) return envBase

  // Same-origin FastAPI server (localhost, Cloud Run, Railway, Cloudflare tunnel)
  return ''
}

export function setApiBase(url) {
  const cleaned = (url || '').trim().replace(/\/+$/, '')
  try {
    if (cleaned) {
      window.localStorage.setItem(STORAGE_KEY, cleaned)
    } else {
      window.localStorage.removeItem(STORAGE_KEY)
    }
  } catch {}
  return cleaned
}

async function request(path, options = {}) {
  const base = getApiBase()
  if (isStaticHost() && !base) {
    throw new Error(
      'Backend API URL not configured for GitHub Pages. Click the API status button in the top-right header and paste your live FastAPI backend URL (Cloudflare Tunnel / Railway / Cloud Run).',
    )
  }
  const res = await fetch(`${base}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`)
  }
  return res.json()
}

export const fetchHealth = () => request('/api/health')

export async function fetchMetrics() {
  try {
    return await request('/api/metrics')
  } catch {
    // Graceful static fallback on GitHub Pages so Holdout Evaluation always renders verified metrics
    const res = await fetch('./holdout_metrics.json')
    if (!res.ok) throw new Error('Unable to load holdout metrics.')
    return res.json()
  }
}

export const fetchBurst = () => request('/api/burst')
export const simulateBurst = () => request('/api/burst/simulate', { method: 'POST' })
export const resetBurst = () => request('/api/burst/reset', { method: 'POST' })

export const submitTriage = ({ subject, body, drafterMode, recordInBurstRadar = true }) =>
  request('/api/triage', {
    method: 'POST',
    body: JSON.stringify({
      subject: subject ?? '',
      body,
      drafter_mode: drafterMode,
      record_in_burst_radar: recordInBurstRadar,
    }),
  })
