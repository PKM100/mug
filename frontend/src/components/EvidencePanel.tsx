import { useMemo, useState, type ReactNode } from 'react'
import type { Candidate, GateResult } from '../api'
import { domId } from '../format'

interface Props {
  pool: Candidate[]
  poolTokens: number | null
  result: GateResult | null
  running: boolean
  closedBookGate: boolean
}

type Filter = 'all' | 'kept' | 'withheld'

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** Render `text` with `needle` highlighted (whitespace-insensitive). Returns null if not found. */
function highlight(text: string, needle: string): ReactNode | null {
  const words = needle.trim().replace(/^["“]|["”]$/g, '').split(/\s+/).filter(Boolean)
  if (words.length === 0) return null
  const re = new RegExp(words.map(escapeRe).join('\\s+'), 'i')
  const m = re.exec(text)
  if (!m) return null
  return (
    <>
      {text.slice(0, m.index)}
      <mark className="decisive">{m[0]}</mark>
      {text.slice(m.index + m[0].length)}
    </>
  )
}

function SourceBadge({ c }: { c: Candidate }) {
  if (c.source === 'pubmed') {
    const bits = [c.meta?.journal, c.meta?.year].filter(Boolean).join(' · ')
    const q = c.meta?.quality
    return (
      <>
        <span className="badge pubmed">
          PubMed{bits && <span className="badge-meta"> · {bits}</span>}
        </span>
        {q && (
          <span
            className={`badge evidence lvl-${q.retracted ? 'x' : Math.min(q.level, 8)}`}
            title={[`Publication types: ${(c.meta?.pub_types ?? []).join(', ') || 'none listed'}`, ...q.flags].join('\n')}
          >
            {q.label}
            {q.flags.includes('older than 15 years') ? ' · old' : ''}
          </span>
        )}
      </>
    )
  }
  if (c.source === 'textbook') return <span className="badge textbook">Textbook</span>
  return <span className="badge user">User</span>
}

function PassageCard({ c, keptAs, decided, quote }: { c: Candidate; keptAs: { doc: number; cand: Candidate } | null; decided: boolean; quote: string | null }) {
  const decisive = keptAs?.cand.full_content ? keptAs.cand.content : null
  const verified = keptAs && quote ? highlight(c.content, quote) : null
  const [expanded, setExpanded] = useState<boolean>(!!decisive || !!verified)
  const hl = verified ?? (decisive ? highlight(c.content, decisive) : null)
  const status = !decided ? '' : keptAs ? 'kept' : 'withheld'
  const long = c.content.length > 420

  return (
    <article id={domId(c.id)} className={`passage ${status}`}>
      <header>
        <SourceBadge c={c} />
        {keptAs && <span className="badge kept">Kept · Doc {keptAs.doc}</span>}
        {verified && <span className="badge verified">Verifier quote highlighted</span>}
        {decided && !keptAs && <span className="badge withheld">Withheld</span>}
        {typeof c.score === 'number' && <span className="muted small score">sim {c.score.toFixed(3)}</span>}
        <span className="muted small pid">{c.id}</span>
      </header>
      <h4>
        {c.url ? (
          <a href={c.url} target="_blank" rel="noreferrer noopener">
            {c.title || c.id} ↗
          </a>
        ) : (
          c.title || c.id
        )}
      </h4>
      {decisive && !hl && (
        <blockquote className="decisive-quote">
          <span className="small">Decisive sentence</span>
          {decisive}
        </blockquote>
      )}
      <p className={`content ${expanded || !long ? '' : 'clamped'}`}>{hl ?? c.content}</p>
      {long && (
        <button type="button" className="link-btn small" onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </article>
  )
}

export function EvidencePanel({ pool, poolTokens, result, running, closedBookGate }: Props) {
  const [filter, setFilter] = useState<Filter>('all')
  const kept = useMemo(() => {
    const m = new Map<string, { doc: number; cand: Candidate }>()
    if (result?.admitted) result.kept_context.forEach((cand, i) => m.set(cand.id, { doc: i + 1, cand }))
    return m
  }, [result])
  const decided = !!result
  // Kept passages first, so the evidence the answer actually used is visible without scrolling.
  const shown = pool
    .filter((c) => filter === 'all' || (filter === 'kept' ? kept.has(c.id) : !kept.has(c.id)))
    .sort((a, b) => Number(kept.has(b.id)) - Number(kept.has(a.id)))
  const quote = result?.verification?.supported ? result.verification.quote : null
  const counts = {
    textbook: pool.filter((c) => c.source === 'textbook').length,
    pubmed: pool.filter((c) => c.source === 'pubmed').length,
    user: pool.filter((c) => c.source === 'user').length,
  }

  return (
    <section className="card evidence">
      <div className="card-title">
        <h2>Evidence pool</h2>
        {pool.length > 0 && (
          <span className="muted small">
            {pool.length} passages · {counts.pubmed} PubMed · {counts.textbook} textbook{counts.user ? ` · ${counts.user} user` : ''}
            {poolTokens != null && ` · ~${poolTokens.toLocaleString()} tokens`}
          </span>
        )}
        {decided && pool.length > 0 && (
          <div className="seg" role="tablist">
            {(['all', 'kept', 'withheld'] as Filter[]).map((f) => (
              <button key={f} type="button" role="tab" aria-selected={filter === f} className={filter === f ? 'on' : ''} onClick={() => setFilter(f)}>
                {f}
                <span className="count">{f === 'all' ? pool.length : f === 'kept' ? kept.size : pool.length - kept.size}</span>
              </button>
            ))}
          </div>
        )}
      </div>
      {pool.length === 0 ? (
        <p className="muted empty">
          {closedBookGate
            ? 'Closed-book gate: nothing is retrieved.'
            : running
              ? 'Retrieving from MedRAG textbooks and PubMed…'
              : decided
                ? 'Retrieval returned no passages.'
                : 'Retrieved passages will appear here.'}
        </p>
      ) : (
        <>
          {decided && !result?.admitted && (
            <p className="field-hint">
              The gate withheld all retrieved evidence — none of it was shown to the final answer. Withheld does not mean the evidence was
              wrong or that the closed-book answer is endorsed.
            </p>
          )}
          <div className="passages-list">
            {shown.map((c) => (
              <PassageCard key={`${c.id}-${kept.has(c.id)}`} c={c} keptAs={kept.get(c.id) ?? null} decided={decided} quote={quote} />
            ))}
            {shown.length === 0 && <p className="muted empty">No passages in this view.</p>}
          </div>
        </>
      )}
    </section>
  )
}
