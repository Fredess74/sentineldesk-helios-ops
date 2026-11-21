import React from 'react'
import { Globe2 } from 'lucide-react'

export default function PrecedentList({ precedents, highlightedTicketId, citedIds = [] }) {
  if (!precedents?.length) return null

  return (
    <div className="rounded-2xl border border-white/[0.08] bg-[#111114] p-5">
      <div className="mb-3.5 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">Top Retrieved Historical Tickets</h3>
        <span className="font-mono text-[11px] text-zinc-500">Top {precedents.length} from kb_train</span>
      </div>

      <div className="divide-y divide-white/[0.06]">
        {precedents.map((p) => {
          const isHighlighted = highlightedTicketId === p.ticket_id
          const isCited = citedIds.includes(p.ticket_id)
          const simPct = Math.round((p.similarity || 0) * 100)

          return (
            <div
              key={p.ticket_id}
              id={`precedent-${p.ticket_id}`}
              className={`-mx-2 rounded-xl border border-transparent px-3 py-3 transition-all ${
                isHighlighted ? 'citation-highlight' : 'hover:bg-white/[0.02]'
              }`}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-md bg-sky-500/15 px-2 py-0.5 font-mono text-xs font-semibold text-sky-300">
                    {p.ticket_id}
                  </span>
                  <span className="font-mono text-[11px] uppercase text-zinc-400">{p.language}</span>
                  {p.cross_lingual && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-indigo-500/15 px-2 py-0.5 text-[11px] font-medium text-indigo-300">
                      <Globe2 className="h-3 w-3" />
                      Cross-lingual
                    </span>
                  )}
                  {isCited && (
                    <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] font-medium text-emerald-300">
                      Cited
                    </span>
                  )}
                  <span className="text-xs text-zinc-500">
                    {p.queue} · {p.type}
                  </span>
                </div>
                <span className="font-mono text-xs font-semibold text-zinc-200">{simPct}% match</span>
              </div>

              <div className="mt-1.5 text-xs font-medium text-zinc-200">{p.subject}</div>
              <p className="mt-1 line-clamp-3 text-xs leading-relaxed text-zinc-400">{p.answer}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
