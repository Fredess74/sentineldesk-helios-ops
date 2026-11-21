import React, { useEffect, useState } from 'react'
import { Flame, Shield } from 'lucide-react'
import {
  fetchBurst,
  fetchHealth,
  fetchMetrics,
  resetBurst,
  simulateBurst,
  submitTriage,
} from './api'
import AuditGlassPanel from './components/AuditGlassPanel'
import HoldoutScorecard from './components/HoldoutScorecard'
import IntakePanel from './components/IntakePanel'
import OutageBanner from './components/OutageBanner'

export default function App() {
  const [activeTab, setActiveTab] = useState('console')
  const [health, setHealth] = useState(null)
  const [burst, setBurst] = useState({ active_outage: false, incidents: [], buffer_size: 0 })
  const [metrics, setMetrics] = useState(null)
  const [metricsError, setMetricsError] = useState(null)

  const [subject, setSubject] = useState('')
  const [body, setBody] = useState('')
  const [drafterMode, setDrafterMode] = useState('cloud')
  const [loading, setLoading] = useState(false)
  const [simulating, setSimulating] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [highlightedTicketId, setHighlightedTicketId] = useState(null)

  useEffect(() => {
    fetchHealth()
      .then((h) => {
        setHealth(h)
        setDrafterMode(h.cloud_drafter_enabled ? 'cloud' : 'local')
      })
      .catch(() => {})

    fetchBurst().then(setBurst).catch(() => {})
    fetchMetrics().then(setMetrics).catch((e) => setMetricsError(e.message))
  }, [])

  const runTriage = async (subjVal, bodyVal, modeOverride) => {
    const useMode = modeOverride || drafterMode
    if (!bodyVal.trim()) return
    setLoading(true)
    setError(null)
    setHighlightedTicketId(null)
    try {
      const data = await submitTriage({
        subject: subjVal,
        body: bodyVal,
        drafterMode: useMode,
        recordInBurstRadar: true,
      })
      setResult(data)
      if (data.burst_radar) setBurst(data.burst_radar)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSelectPreset = (preset) => {
    setSubject(preset.subject)
    setBody(preset.body)
    runTriage(preset.subject, preset.body)
  }

  const handleSimulateBurst = async () => {
    setSimulating(true)
    setError(null)
    try {
      const data = await simulateBurst()
      if (data.burst_radar) setBurst(data.burst_radar)
    } catch (e) {
      setError(e.message)
    } finally {
      setSimulating(false)
    }
  }

  const handleResetBurst = async () => {
    try {
      const data = await resetBurst()
      setBurst(data)
    } catch (e) {
      setError(e.message)
    }
  }

  const handleCiteClick = (ticketId) => {
    setHighlightedTicketId(ticketId)
    const el = document.getElementById(`precedent-${ticketId}`)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setTimeout(() => setHighlightedTicketId((cur) => (cur === ticketId ? null : cur)), 3000)
  }

  return (
    <div className="min-h-screen text-zinc-100">
      {/* Clean Minimal Top Bar */}
      <header className="sticky top-0 z-30 border-b border-white/[0.07] bg-[#09090b]/85 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1360px] items-center justify-between gap-4 px-4 sm:px-6">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white text-zinc-950">
              <Shield className="h-4 w-4" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-sm font-semibold tracking-tight text-white">SentinelDesk</span>
              <span className="hidden text-xs text-zinc-500 sm:inline">/ Helios Managed Ops</span>
            </div>
          </div>

          <nav className="flex items-center rounded-lg border border-white/[0.07] bg-[#111114] p-0.5">
            <button
              type="button"
              onClick={() => setActiveTab('console')}
              className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                activeTab === 'console' ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Live Triage
            </button>
            <button
              type="button"
              onClick={() => {
                setActiveTab('scorecard')
                if (!metrics) {
                  fetchMetrics()
                    .then((m) => {
                      setMetrics(m)
                      setMetricsError(null)
                    })
                    .catch((e) => setMetricsError(e.message))
                }
              }}
              className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                activeTab === 'scorecard' ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Holdout Evaluation
            </button>
          </nav>

          <div className="flex items-center gap-2.5">
            {health && (
              <span className="hidden items-center gap-1.5 font-mono text-[11px] text-zinc-400 md:inline-flex">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                {health.kb_indexed_tickets?.toLocaleString()} KB
              </span>
            )}
            <button
              type="button"
              onClick={burst?.active_outage ? handleResetBurst : handleSimulateBurst}
              disabled={simulating}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition ${
                burst?.active_outage
                  ? 'border-rose-500/40 bg-rose-500/15 text-rose-200 hover:bg-rose-500/25'
                  : 'border-white/[0.08] bg-[#111114] text-zinc-300 hover:border-white/20 hover:text-white'
              }`}
            >
              <Flame className={`h-3.5 w-3.5 ${burst?.active_outage ? 'text-rose-400' : 'text-zinc-400'}`} />
              {simulating
                ? 'Simulating...'
                : burst?.active_outage
                  ? 'Reset Outage'
                  : 'Simulate Outage'}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto flex max-w-[1360px] flex-col gap-5 px-4 py-6 sm:px-6">
        {error && (
          <div className="rounded-xl border border-rose-500/35 bg-rose-950/30 px-4 py-2.5 text-xs text-rose-200">
            {error}
          </div>
        )}

        {activeTab === 'console' ? (
          <>
            <OutageBanner burst={burst} onReset={handleResetBurst} />

            <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
              <div className="lg:col-span-5 xl:col-span-4">
                <IntakePanel
                  subject={subject}
                  body={body}
                  drafterMode={drafterMode}
                  loading={loading}
                  onSubjectChange={setSubject}
                  onBodyChange={setBody}
                  onDrafterModeChange={setDrafterMode}
                  onSelectPreset={handleSelectPreset}
                  onSubmit={() => runTriage(subject, body)}
                />
              </div>

              <div className="lg:col-span-7 xl:col-span-8">
                <AuditGlassPanel
                  result={result}
                  loading={loading}
                  highlightedTicketId={highlightedTicketId}
                  onCiteClick={handleCiteClick}
                />
              </div>
            </div>
          </>
        ) : (
          <HoldoutScorecard metrics={metrics} error={metricsError} />
        )}
      </main>
    </div>
  )
}
