const API_BASE = ''

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
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
export const fetchMetrics = () => request('/api/metrics')
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
