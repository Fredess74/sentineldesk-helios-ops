import React from 'react'
import { Flame, Globe2, RotateCcw } from 'lucide-react'

export default function OutageBanner({ burst, onReset }) {
  const active = burst?.active_outage && burst?.incidents?.length > 0
  if (!active) return null

  const top = burst.incidents[0]

  return (
    <div className="rounded-xl border border-rose-500/40 bg-rose-950/35 px-4 py-3.5 backdrop-blur-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-rose-500/20 text-rose-400">
            <Flame className="h-4 w-4" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded bg-rose-500 px-2 py-0.5 font-mono text-[11px] font-semibold uppercase tracking-wide text-white">
                Major Incident Outage
              </span>
              <span className="font-mono text-xs font-medium text-rose-200">{top.master_incident_id}</span>
              <span className="text-xs text-rose-200/80">· {top.ticket_count} correlated tickets</span>
              <span className="inline-flex items-center gap-1 rounded-full border border-rose-400/30 bg-rose-500/10 px-2 py-0.5 font-mono text-[11px] text-rose-200">
                <Globe2 className="h-3 w-3" />
                {top.affected_languages.map((l) => l.toUpperCase()).join(' + ')}
                {top.is_cross_lingual ? ' Cross-Lingual' : ''}
              </span>
              <span className="font-mono text-[11px] text-rose-300/75">
                {(top.avg_cluster_similarity * 100).toFixed(0)}% match
              </span>
            </div>
            <p className="mt-1 text-sm text-rose-100">
              <span className="font-medium text-white">{top.primary_queue}:</span> {top.summary}
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onReset}
          className="inline-flex items-center gap-1.5 rounded-lg border border-rose-400/30 bg-rose-950/60 px-3 py-1.5 text-xs font-medium text-rose-200 transition hover:bg-rose-900/60 hover:text-white"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Clear Alert
        </button>
      </div>

      {top.tickets?.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5 border-t border-rose-500/20 pt-2.5">
          {top.tickets.map((t) => (
            <span
              key={t.request_id}
              className="inline-flex items-center gap-1.5 rounded-md border border-rose-500/25 bg-black/25 px-2 py-1 text-xs text-rose-100"
            >
              <span className="font-mono text-[11px] text-rose-300">{t.request_id}</span>
              <span className="rounded bg-white/10 px-1 font-mono text-[10px] uppercase">{t.language}</span>
              <span className="max-w-[220px] truncate">{t.subject}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
