import React from 'react'
import { ArrowRight, Cloud, Cpu, Lock } from 'lucide-react'
import { DEMO_PRESETS } from '../presets'

export default function IntakePanel({
  subject,
  body,
  drafterMode,
  loading,
  onSubjectChange,
  onBodyChange,
  onDrafterModeChange,
  onSelectPreset,
  onSubmit,
}) {
  return (
    <div className="flex flex-col gap-5 rounded-2xl border border-white/[0.08] bg-[#111114] p-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white">Incoming Ticket</h2>
          <p className="mt-0.5 text-xs text-zinc-400">
            Paste any unseen English or German ticket to triage in real time.
          </p>
        </div>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          onSubmit()
        }}
        className="flex flex-col gap-3.5"
      >
        <div>
          <label className="mb-1.5 flex items-center justify-between text-xs font-medium text-zinc-300">
            <span>Subject</span>
            <span className="text-[11px] text-zinc-500">Optional</span>
          </label>
          <input
            type="text"
            value={subject}
            onChange={(e) => onSubjectChange(e.target.value)}
            placeholder="Leave empty to test null-subject handling"
            className="w-full rounded-xl border border-white/[0.08] bg-[#09090b] px-3.5 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 outline-none transition focus:border-sky-400/60 focus:ring-2 focus:ring-sky-500/15"
          />
        </div>

        <div>
          <label className="mb-1.5 flex items-center justify-between text-xs font-medium text-zinc-300">
            <span>Message Body</span>
            <span className="font-mono text-[11px] text-zinc-500">{body.length} chars</span>
          </label>
          <textarea
            rows={7}
            value={body}
            onChange={(e) => onBodyChange(e.target.value)}
            placeholder="Describe the incident, error message, billing question, or request..."
            className="w-full resize-y rounded-xl border border-white/[0.08] bg-[#09090b] px-3.5 py-2.5 text-sm leading-relaxed text-zinc-100 placeholder-zinc-600 outline-none transition focus:border-sky-400/60 focus:ring-2 focus:ring-sky-500/15"
          />
        </div>

        {/* Clean Privacy Mode Toggle */}
        <div className="flex items-center justify-between rounded-xl border border-white/[0.06] bg-[#09090b] p-1">
          <button
            type="button"
            onClick={() => onDrafterModeChange('cloud')}
            className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium transition ${
              drafterMode === 'cloud'
                ? 'bg-zinc-800 text-white shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Cloud className="h-3.5 w-3.5 text-sky-400" />
            Cloud · PII-Scrubbed
          </button>
          <button
            type="button"
            onClick={() => onDrafterModeChange('local')}
            className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium transition ${
              drafterMode === 'local'
                ? 'bg-zinc-800 text-white shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Cpu className="h-3.5 w-3.5 text-emerald-400" />
            Local · Air-Gapped
          </button>
        </div>

        <button
          type="submit"
          disabled={loading || !body.trim()}
          className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-45"
        >
          {loading ? 'Analyzing Ticket...' : 'Triage & Draft Reply'}
          {!loading && <ArrowRight className="h-4 w-4" />}
        </button>
      </form>

      {/* Compact 2x2 Sample Presets */}
      <div className="border-t border-white/[0.07] pt-4">
        <div className="mb-2.5 text-xs font-medium text-zinc-400">Quick Test Scenarios</div>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {DEMO_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              disabled={loading}
              onClick={() => onSelectPreset(preset)}
              className="group rounded-xl border border-white/[0.07] bg-[#09090b]/80 px-3 py-2.5 text-left transition hover:border-white/20 hover:bg-zinc-900 disabled:opacity-50"
            >
              <div className="font-mono text-[10px] font-medium uppercase tracking-wider text-zinc-500 group-hover:text-sky-400">
                {preset.tag}
              </div>
              <div className="mt-0.5 text-xs font-medium text-zinc-200 group-hover:text-white">
                {preset.title}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-2 text-[11px] text-zinc-500">
        <Lock className="h-3.5 w-3.5 shrink-0 text-emerald-500/80" />
        <span>Local PII/PHI vault scrubs emails, IPs, phones, IBANs &amp; IDs before any cloud call.</span>
      </div>
    </div>
  )
}
