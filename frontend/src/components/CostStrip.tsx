import type { GateResult, Span, Summary } from '../api'
import { fmtInt, fmtMs } from '../format'

interface Props {
  result: GateResult
  summary: Summary
  spans: Span[]
}

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: 'good' | 'warn' }) {
  return (
    <div className={`stat ${tone ?? ''}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

export function CostStrip({ result, summary, spans }: Props) {
  const pool = result.pool_tokens
  const kept = result.retained_tokens
  const saved = pool > 0 ? 1 - kept / pool : null
  const speedup = summary.wall_ms > 0 ? summary.sequential_ms / summary.wall_ms : null
  const ratio = pool > 0 ? Math.min(1, kept / pool) : 0
  const byModel = new Map<string, number>()
  for (const sp of spans) if (sp.kind === 'llm' && sp.model) byModel.set(sp.model, (byModel.get(sp.model) ?? 0) + sp.prompt_tokens)

  return (
    <section className="card cost">
      <div className="stat wide">
        <div className="stat-label">Context retained vs. naive pool</div>
        <div className="stat-value">
          {fmtInt(kept)} <span className="of">/ {fmtInt(pool)} tok</span>
          {saved != null && <span className={`saved ${saved > 0 ? 'good' : ''}`}>{Math.round(saved * 100)}% saved</span>}
        </div>
        <div className="token-bar" aria-hidden>
          <span style={{ width: `${ratio * 100}%` }} />
        </div>
      </div>
      <Stat label="LLM calls" value={String(summary.llm_calls)} sub={summary.llm_calls_used !== summary.llm_calls ? `${summary.llm_calls_used} used` : 'all used'} />
      <Stat label="Prompt tokens" value={fmtInt(summary.prompt_tokens)} sub={`${fmtInt(summary.completion_tokens)} completion`} />
      {byModel.size > 1 && (
        <Stat
          label="Prompt tokens by model"
          value={[...byModel.values()].map(fmtInt).join(' / ')}
          sub={[...byModel.keys()].join(' / ')}
        />
      )}
      <Stat
        label="Wall time"
        value={fmtMs(summary.wall_ms)}
        sub={speedup != null ? `vs ${fmtMs(summary.sequential_ms)} sequential · ${speedup.toFixed(2)}×` : undefined}
        tone={speedup != null && speedup > 1.05 ? 'good' : undefined}
      />
      <Stat
        label="Wasted speculative"
        value={`${fmtInt(summary.wasted_prompt_tokens)} tok`}
        sub={summary.wasted_prompt_tokens > 0 ? 'discarded evaluator call' : 'none'}
        tone={summary.wasted_prompt_tokens > 0 ? 'warn' : undefined}
      />
    </section>
  )
}
