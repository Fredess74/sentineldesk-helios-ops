import React, { useEffect, useState } from 'react'
import {
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Copy,
  Lock,
  ShieldAlert,
  ShieldCheck,
  Siren,
  Sparkles,
} from 'lucide-react'
import GroundingAuditCard from './GroundingAuditCard'
import PiiDiffDrawer from './PiiDiffDrawer'
import PrecedentList from './PrecedentList'

function renderCitations(text, onCiteClick) {
  if (!text) return null
  return text.split(/(\[TKT-\d{5}\])/g).map((part, idx) => {
    const m = part.match(/^\[(TKT-\d{5})\]$/)
    if (m) {
      const tid = m[1]
      return (
        <button
          key={idx}
          type="button"
          onClick={() => onCiteClick(tid)}
          title={`Jump to precedent ${tid}`}
          className="mx-0.5 inline-flex items-center rounded bg-sky-500/15 px-1.5 py-0.5 font-mono text-xs font-semibold text-sky-300 transition hover:bg-sky-500/30 hover:text-white"
        >
          [{tid}]
        </button>
      )
    }
    return <React.Fragment key={idx}>{part}</React.Fragment>
  })
}

function MetricTile({ label, value, confidence, accent = 'zinc' }) {
  const pct = Math.round((confidence ?? 0) * 100)
  const low = confidence < 0.65
  const textTone =
    accent === 'rose'
      ? 'text-rose-300'
      : accent === 'amber'
        ? 'text-amber-300'
        : accent === 'emerald'
          ? 'text-emerald-300'
          : 'text-white'

  return (
    <div className="rounded-xl border border-white/[0.07] bg-[#09090b] px-4 py-3">
      <div className="flex items-center justify-between text-xs text-zinc-500">
        <span>{label}</span>
        <span className={`font-mono font-medium ${low ? 'text-amber-400' : 'text-zinc-300'}`}>
          {pct}%
        </span>
      </div>
      <div className={`mt-1 truncate text-sm font-semibold ${textTone}`}>{value}</div>
      <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-zinc-800">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            low ? 'bg-amber-400' : 'bg-sky-400'
          }`}
          style={{ width: `${Math.max(5, Math.min(100, pct))}%` }}
        />
      </div>
    </div>
  )
}

export default function AuditGlassPanel({ result, loading, highlightedTicketId, onCiteClick }) {
  const [copied, setCopied] = useState(false)
  const [showAudit, setShowAudit] = useState(false)
  const [showPii, setShowPii] = useState(false)

  useEffect(() => {
    setCopied(false)
    setShowAudit(Boolean(result?.grounding_audit?.veto_triggered))
    setShowPii(Boolean(result?.privacy_vault?.pii_detected))
  }, [result?.request_id])

  if (loading) {
    return (
      <div className="flex min-h-[480px] flex-col items-center justify-center rounded-2xl border border-white/[0.08] bg-[#111114] p-10 text-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-zinc-700 border-t-sky-400" />
        <p className="mt-4 text-sm font-medium text-zinc-200">Triaging ticket &amp; retrieving precedents...</p>
        <p className="mt-1 text-xs text-zinc-500">PII vault · EmbeddingGemma 2 · Calibrated ensemble · Grounded draft</p>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="flex min-h-[480px] flex-col items-center justify-center rounded-2xl border border-dashed border-white/[0.1] bg-[#111114]/60 p-10 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/[0.08] bg-zinc-900 text-sky-400">
          <Sparkles className="h-5 w-5" />
        </div>
        <h3 className="mt-4 text-base font-semibold text-white">Ready to Triage</h3>
        <p className="mt-1.5 max-w-md text-sm text-zinc-400">
          Paste any English or German ticket on the left — or choose a quick test scenario — to inspect routing predictions, escalation status, retrieved precedents, and a grounded reply.
        </p>
      </div>
    )
  }

  const { triage, escalation, resolution, grounding_audit, privacy_vault, precedents, latency_ms } = result
  const isEscalated = escalation.status === 'ESCALATE_ON_CALL'
  const prioTone =
    triage.priority.label === 'high'
      ? 'rose'
      : triage.priority.label === 'medium'
        ? 'amber'
        : 'emerald'

  const copyReply = () => {
    const txt = resolution?.draft_reply || resolution?.safe_holding_ack || ''
    if (!txt) return
    navigator.clipboard?.writeText(txt).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    })
  }

  return (
    <div className="flex flex-col gap-4">
      {/* 1. Decision + Triage Summary Card */}
      <div className="rounded-2xl border border-white/[0.08] bg-[#111114] p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2.5">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${
                isEscalated
                  ? 'border border-rose-500/40 bg-rose-500/15 text-rose-300'
                  : 'border border-emerald-500/40 bg-emerald-500/15 text-emerald-300'
              }`}
            >
              {isEscalated ? <Siren className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
              {isEscalated ? 'Escalate to On-Call Engineer' : 'Auto-Draft Ready'}
            </span>

            <span className="rounded-full border border-white/[0.08] bg-zinc-900 px-2.5 py-0.5 font-mono text-[11px] text-zinc-300">
              {result.detected_language.toUpperCase()}
            </span>

            {result.subject_missing && (
              <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-0.5 text-[11px] font-medium text-amber-300">
                No Subject Handled
              </span>
            )}
          </div>

          <div className="flex items-center gap-1.5 font-mono text-xs text-zinc-400" title={`Privacy ${latency_ms.privacy_ms}ms · Retrieval ${latency_ms.retrieval_ms}ms · Triage ${latency_ms.triage_ms}ms · Draft ${latency_ms.draft_ms}ms · Audit ${latency_ms.audit_ms}ms`}>
            <Clock className="h-3.5 w-3.5 text-zinc-500" />
            <span className="font-semibold text-zinc-200">{latency_ms.total_ms} ms</span>
            <span className="hidden text-zinc-500 sm:inline">
              (triage {Math.round(latency_ms.triage_ms + latency_ms.retrieval_ms)} ms)
            </span>
          </div>
        </div>

        {/* Concise Escalation Reasons (Only when escalated) */}
        {isEscalated && escalation.reasons?.length > 0 && (
          <div className="mt-3 rounded-xl border border-rose-500/25 bg-rose-950/20 px-3.5 py-2.5 text-xs text-rose-200">
            <ul className="space-y-1">
              {escalation.reasons.map((r, i) => (
                <li key={i}>
                  <span className="font-semibold text-rose-300">{r.label}:</span> {r.detail}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* 3 Clean Routing Cards */}
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricTile
            label="Predicted Type"
            value={triage.type.label}
            confidence={triage.type.confidence}
            accent={triage.type.label === 'Incident' ? 'rose' : 'zinc'}
          />
          <MetricTile
            label="Target Queue"
            value={triage.queue.label}
            confidence={triage.queue.confidence}
          />
          <MetricTile
            label="Priority"
            value={triage.priority.label.toUpperCase()}
            confidence={triage.priority.confidence}
            accent={prioTone}
          />
        </div>
      </div>

      {/* 2. Hero Generated Response Card */}
      {!isEscalated && resolution?.draft_reply ? (
        <div className="rounded-2xl border border-white/[0.08] bg-[#111114] p-5">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Generated Draft Response ({result.detected_language.toUpperCase()})
              </h3>
              <p className="text-xs text-zinc-500">
                Grounded in historical precedents · click any [TKT-XXXXX] citation to inspect
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="rounded-md bg-zinc-900 px-2 py-1 font-mono text-[11px] text-zinc-400">
                {resolution.drafter_mode_used}
              </span>
              <button
                type="button"
                onClick={copyReply}
                className="inline-flex items-center gap-1.5 rounded-lg border border-white/[0.08] bg-zinc-900 px-2.5 py-1 text-xs font-medium text-zinc-200 transition hover:bg-zinc-800"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
          </div>

          <div className="whitespace-pre-wrap rounded-xl border border-white/[0.06] bg-[#09090b] p-4 text-sm leading-relaxed text-zinc-100">
            {renderCitations(resolution.draft_reply, onCiteClick)}
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="rounded-2xl border border-white/[0.08] bg-[#111114] p-5">
            <div className="mb-2.5 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">
                Safe Holding Reply ({result.detected_language.toUpperCase()})
              </h3>
              <button
                type="button"
                onClick={copyReply}
                className="inline-flex items-center gap-1 rounded-lg border border-white/[0.08] bg-zinc-900 px-2 py-1 text-[11px] font-medium text-zinc-300 hover:bg-zinc-800"
              >
                {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
            <p className="rounded-xl border border-amber-500/20 bg-amber-950/10 p-3.5 text-sm leading-relaxed text-zinc-200">
              {resolution?.safe_holding_ack}
            </p>
          </div>

          <div className="rounded-2xl border border-white/[0.08] bg-[#111114] p-5">
            <h3 className="mb-2.5 text-sm font-semibold text-white">On-Call Engineer Brief</h3>
            {resolution?.oncall_war_room_brief && (
              <div className="space-y-3 text-xs">
                <div>
                  <div className="text-[11px] font-medium text-zinc-500">Suspected System</div>
                  <div className="mt-0.5 font-medium text-zinc-100">
                    {resolution.oncall_war_room_brief.suspected_system}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-medium text-zinc-500">Relevant Historical Precedents</div>
                  <ul className="mt-1 space-y-1 text-zinc-300">
                    {resolution.oncall_war_room_brief.historical_root_causes.slice(0, 2).map((rc, i) => (
                      <li key={i} className="leading-relaxed">
                        {renderCitations(rc, onCiteClick)}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <div className="text-[11px] font-medium text-zinc-500">Immediate Diagnostic Checks</div>
                  <ul className="mt-1 list-disc space-y-1 pl-4 text-zinc-400">
                    {resolution.oncall_war_room_brief.diagnostic_questions.slice(0, 2).map((dq, i) => (
                      <li key={i}>{dq}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 3. Top Retrieved Historical Precedents */}
      <PrecedentList
        precedents={precedents}
        highlightedTicketId={highlightedTicketId}
        citedIds={resolution?.cited_ticket_ids || []}
      />

      {/* 4. Clean Collapsible Guardrail Inspectors (Zero Clutter by Default) */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setShowAudit((v) => !v)}
            className={`inline-flex items-center gap-2 rounded-xl border px-3.5 py-2 text-xs font-medium transition ${
              grounding_audit?.veto_triggered
                ? 'border-rose-500/40 bg-rose-500/15 text-rose-200'
                : 'border-white/[0.08] bg-[#111114] text-zinc-300 hover:border-white/20'
            }`}
          >
            {grounding_audit?.veto_triggered ? (
              <ShieldAlert className="h-3.5 w-3.5 text-rose-400" />
            ) : (
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
            )}
            Grounding Audit: {grounding_audit?.veto_triggered ? 'Policy Vetoed' : 'Verified'}
            {showAudit ? <ChevronUp className="h-3.5 w-3.5 opacity-60" /> : <ChevronDown className="h-3.5 w-3.5 opacity-60" />}
          </button>

          <button
            type="button"
            onClick={() => setShowPii((v) => !v)}
            className={`inline-flex items-center gap-2 rounded-xl border px-3.5 py-2 text-xs font-medium transition ${
              privacy_vault?.pii_detected
                ? 'border-emerald-500/35 bg-emerald-500/10 text-emerald-200'
                : 'border-white/[0.08] bg-[#111114] text-zinc-300 hover:border-white/20'
            }`}
          >
            <Lock className="h-3.5 w-3.5 text-emerald-400" />
            PII Vault:{' '}
            {privacy_vault?.pii_detected
              ? `${privacy_vault.redaction_count} Scrubbed`
              : '0 Detected'}
            {showPii ? <ChevronUp className="h-3.5 w-3.5 opacity-60" /> : <ChevronDown className="h-3.5 w-3.5 opacity-60" />}
          </button>

          <span className="ml-auto font-mono text-[11px] text-zinc-500">
            Engine: {triage.decision_engine_mode} · Agreement: {triage.model_agreement?.status}
          </span>
        </div>

        {showAudit && <GroundingAuditCard audit={grounding_audit} />}
        {showPii && <PiiDiffDrawer privacyVault={privacy_vault} />}
      </div>
    </div>
  )
}
