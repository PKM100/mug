import type { Span } from '../api'
import { fmtInt, fmtMs } from '../format'

interface Props {
  spans: Span[]
  running: boolean
  elapsedMs: number
  wallMs: number | null
}

function niceStep(total: number): number {
  const raw = total / 6
  const pow = 10 ** Math.floor(Math.log10(Math.max(raw, 1)))
  for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * pow) return m * pow
  return 10 * pow
}

function describe(s: Span): string {
  const lines = [`${s.name} — ${fmtMs(s.duration_ms)}`, `t = ${fmtMs(s.start_ms)} → ${fmtMs(s.end_ms)}`]
  if (s.model) lines.push(`model: ${s.model}`)
  if (s.kind === 'llm') lines.push(`tokens: ${s.prompt_tokens} in / ${s.completion_tokens} out`)
  if (s.speculative) lines.push(s.used ? 'speculative (used)' : 'speculative — discarded')
  for (const [k, v] of Object.entries(s.detail ?? {})) {
    lines.push(`${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
  }
  return lines.join('\n')
}

function detailChip(s: Span): string | null {
  const d = s.detail ?? {}
  if (s.kind === 'retrieval') return `${d.n ?? 0} hits${d.error ? ' · error' : ''}`
  if (s.kind === 'decision') {
    if ('moved' in d) return d.moved ? 'moved ✓' : 'no move'
    if ('groundedness' in d) return `g=${Number(d.groundedness).toFixed(2)}`
  }
  if (s.kind === 'llm') return `${fmtInt(s.prompt_tokens)} tok`
  return null
}

export function Timeline({ spans, running, elapsedMs, wallMs }: Props) {
  const sorted = [...spans].sort((a, b) => a.start_ms - b.start_ms || a.end_ms - b.end_ms)
  const maxEnd = sorted.reduce((m, s) => Math.max(m, s.end_ms), 0)
  const total = Math.max(running ? elapsedMs : (wallMs ?? 0), maxEnd, 1)
  const step = niceStep(total)
  const ticks: number[] = []
  for (let t = 0; t <= total + 1e-6; t += step) ticks.push(t)
  const pct = (ms: number) => `${(ms / total) * 100}%`

  return (
    <section className="card timeline-card">
      <div className="card-title">
        <h2>Pipeline trace</h2>
        <span className={`elapsed ${running ? 'live' : ''}`}>
          {running && <i className="pulse" />}
          {fmtMs(running ? elapsedMs : (wallMs ?? maxEnd))}
          {running && <span className="muted small">elapsed</span>}
        </span>
        <div className="legend">
          <span className="lg lg-llm">LLM call</span>
          <span className="lg lg-retrieval">Retrieval</span>
          <span className="lg lg-decision">Decision</span>
          <span className="lg lg-spec">Speculative</span>
          <span className="lg lg-unused">Discarded</span>
        </div>
      </div>

      {sorted.length === 0 ? (
        <p className="muted empty">{running ? 'Waiting for the first span to finish…' : 'Spans appear here as each step completes.'}</p>
      ) : (
        <div className="gantt">
          <div className="gantt-axis">
            <div className="gantt-label" />
            <div className="gantt-track">
              {ticks.map((t) => (
                <span key={t} className="tick" style={{ left: pct(t) }}>
                  {fmtMs(t)}
                </span>
              ))}
            </div>
          </div>
          {sorted.map((s, i) => {
            const cls = [`k-${s.kind}`, s.speculative ? 'spec' : '', s.used ? '' : 'unused'].join(' ')
            const nearEnd = s.end_ms / total > 0.78
            const chip = detailChip(s)
            return (
              <div className={`gantt-row ${cls}`} key={`${s.name}-${s.start_ms}-${i}`} title={describe(s)}>
                <div className="gantt-label">
                  <span className="span-name">{s.name}</span>
                  {s.model && <span className="span-model">{s.model}</span>}
                </div>
                <div className="gantt-track">
                  {ticks.map((t) => (
                    <span key={t} className="grid" style={{ left: pct(t) }} />
                  ))}
                  {s.kind === 'decision' ? (
                    <span className="diamond" style={{ left: pct(s.end_ms) }} />
                  ) : (
                    <span className="bar" style={{ left: pct(s.start_ms), width: `max(3px, ${pct(s.duration_ms)})` }} />
                  )}
                  <span
                    className={`bar-meta ${nearEnd ? 'before' : ''}`}
                    style={nearEnd ? { right: `calc(${pct(total - s.start_ms)} + 8px)` } : { left: `calc(${pct(s.end_ms)} + 8px)` }}
                  >
                    {s.kind !== 'decision' && (s.detail?.timing ? 'untimed (recorded separately)' : fmtMs(s.duration_ms))}
                    {chip && <em>{s.kind !== 'decision' ? ' · ' : ''}{chip}</em>}
                  </span>
                </div>
              </div>
            )
          })}
          {running && (
            <div className="gantt-overlay" aria-hidden>
              <span className="now" style={{ left: pct(elapsedMs) }} />
            </div>
          )}
        </div>
      )}
    </section>
  )
}
