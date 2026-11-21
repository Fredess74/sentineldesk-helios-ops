import React from 'react'
import { Lock } from 'lucide-react'

function highlightTokens(text) {
  if (!text) return null
  return text.split(/(\[[A-Z0-9_]+_\d+\])/g).map((part, idx) =>
    /^\[[A-Z0-9_]+_\d+\]$/.test(part) ? (
      <span
        key={idx}
        className="mx-0.5 inline-block rounded bg-emerald-500/20 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-emerald-300"
      >
        {part}
      </span>
    ) : (
      <React.Fragment key={idx}>{part}</React.Fragment>
    ),
  )
}

export default function PiiDiffDrawer({ privacyVault }) {
  if (!privacyVault) return null

  return (
    <div className="rounded-xl border border-white/[0.08] bg-[#09090b] p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs font-semibold text-zinc-200">
          <Lock className="h-4 w-4 text-emerald-400" />
          What Left Your Network — Local PII/PHI Redaction Diff
        </div>
        <span className="font-mono text-[11px] text-emerald-400">
          {privacyVault.pii_detected
            ? `${privacyVault.redaction_count} entities masked locally · 0 raw PII sent`
            : '0 PII entities detected'}
        </span>
      </div>

      {privacyVault.entities_found?.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-1.5">
          {privacyVault.entities_found.map((e, idx) => (
            <span
              key={idx}
              className="rounded-md border border-emerald-500/25 bg-emerald-500/10 px-2 py-0.5 font-mono text-[11px] text-emerald-300"
            >
              {e.placeholder} <span className="text-zinc-400">({e.preview})</span>
            </span>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="rounded-lg border border-white/[0.06] bg-[#111114] p-3">
          <div className="mb-1.5 text-[11px] font-medium text-zinc-400">Original (Stays Local)</div>
          <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-zinc-300">
            {privacyVault.original_text}
          </pre>
        </div>
        <div className="rounded-lg border border-emerald-500/20 bg-emerald-950/15 p-3">
          <div className="mb-1.5 text-[11px] font-medium text-emerald-300">Scrubbed Payload Sent to LLM</div>
          <div className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-zinc-200">
            {highlightTokens(privacyVault.redacted_text)}
          </div>
        </div>
      </div>
    </div>
  )
}
