import React from 'react'
import { ShieldAlert, ShieldCheck } from 'lucide-react'

function Bar({ label, value, invert = false, threshold }) {
  const pct = Math.round((value ?? 0) * 100)
  const bad = invert ? value > threshold : value < threshold
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="text-zinc-400">{label}</span>
        <span className={`font-mono font-medium ${bad ? 'text-rose-400' : 'text-zinc-200'}`}>{pct}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            bad ? 'bg-rose-500' : 'bg-emerald-500'
          }`}
          style={{ width: `${Math.max(3, Math.min(100, pct))}%` }}
        />
      </div>
    </div>
  )
}

export default function GroundingAuditCard({ audit }) {
  if (!audit) return null
  const scores = audit.scores || {}
  const vetoed = Boolean(audit.veto_triggered)

  return (
    <div
      className={`rounded-xl border p-4 ${
        vetoed ? 'border-rose-500/40 bg-rose-950/20' : 'border-white/[0.08] bg-[#09090b]'
      }`}
    >
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs font-semibold text-zinc-200">
          {vetoed ? (
            <ShieldAlert className="h-4 w-4 text-rose-400" />
          ) : (
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
          )}
          Pass-2 Grounding &amp; Policy Verification
        </div>
        <span className="font-mono text-[11px] text-zinc-400">
          {audit.auditor_mode}
          {audit.rapidfuzz_grounding_score !== undefined &&
            ` · RapidFuzz ${Math.round(audit.rapidfuzz_grounding_score * 100)}%`}
          {' · '}Citations {audit.citation_valid ? 'Valid' : 'Invalid'} · Policy{' '}
          {audit.deterministic_policy_passed ? 'Pass' : 'Blocked'}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Bar label="Claims supported by precedents (min 85%)" value={scores.all_claims_supported} threshold={0.85} />
        <Bar label="Precedents resolve issue" value={scores.precedents_resolve_issue} threshold={0.6} />
        <Bar label="Fabricated refund / SLA risk (max 15%)" value={scores.fabricates_refund_or_sla} invert threshold={0.15} />
        <Bar label="Correct native language" value={scores.correct_native_language} threshold={0.8} />
      </div>

      {vetoed && audit.veto_reasons?.length > 0 && (
        <div className="mt-3 rounded-lg border border-rose-500/30 bg-rose-950/40 px-3 py-2 text-xs text-rose-200">
          {audit.veto_reasons.join(' ')}
        </div>
      )}
    </div>
  )
}
