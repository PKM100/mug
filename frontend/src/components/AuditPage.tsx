import { Fragment, useCallback, useEffect, useState } from 'react'
import {
  fetchAudit,
  fetchAuditRecord,
  fetchReplayAudit,
  type AuditItem,
  type AuditList,
  type AuditOutcome,
  type AuditRecord,
  type ReplayAudit,
} from '../api'

const OUTCOMES: { key: AuditOutcome; label: string; hint: string }[] = [
  {
    key: 'unverified',
    label: 'Moved, no verifiable quote',
    hint: 'Documents changed the answer but the verifier returned no quote found in a kept passage. Gate mode kept the closed-book answer; shadow mode only records this label.',
  },
  {
    key: 'supported',
    label: 'Quote found + judged supportive',
    hint: 'Documents changed the answer; the verifier model judged a kept passage supportive and its quote occurs verbatim there. An LLM judgement, not a clinical-correctness check.',
  },
  { key: 'admitted', label: 'Moved, admitted (no verifier)', hint: 'Documents changed the answer; no verifier was configured' },
  { key: 'no_effect', label: 'No effect', hint: 'Documents did not change the answer. This does not mean the answer is endorsed or correct.' },
  { key: 'no_documents', label: 'No documents', hint: 'Request carried no documents' },
  { key: 'failed', label: 'Failed', hint: 'A model call failed after retries, so the label is unknown' },
]

function Detail({ item, replay }: { item: AuditItem; replay: ReplayAudit | null }) {
  const [rec, setRec] = useState<AuditRecord | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    if (replay) {
      const r = replay.records[item.id]
      if (r) setRec(r)
      else setErr('Record missing from the replay bundle')
      return
    }
    fetchAuditRecord(item.id).then(setRec, (e: Error) => setErr(e.message))
  }, [item.id, replay])
  if (err) return <div className="alert error">{err}</div>
  if (!rec) return <div className="muted small">Loading…</div>
  return (
    <div className="audit-detail">
      <div className="audit-answers">
        <div><span className="eyebrow">Plain RAG would say</span><p>{item.rag_answer ?? '—'}</p></div>
        <div><span className="eyebrow">Closed-book</span><p>{item.closed_book_answer ?? '—'}</p></div>
        <div>
          <span className="eyebrow">{item.proxy_mode === 'shadow' ? 'Returned (shadow: plain RAG, unchanged)' : 'MUG returned'}</span>
          <p>{item.answer}</p>
          {item.proxy_mode === 'shadow' && item.gate_answer && <p className="muted small">Gate would have returned: {item.gate_answer}</p>}
        </div>
      </div>
      {item.quote && (
        <blockquote className="audit-quote">“{item.quote}”</blockquote>
      )}
      <p className="muted small">{rec.result.note}</p>
      {item.outcome === 'no_effect' || item.outcome === 'unverified' ? (
        <p className="withheld-hint">
          <b>Withheld is not endorsed.</b> The closed-book answer was not checked against the literature.
        </p>
      ) : null}
      {rec.failed_calls && rec.failed_calls.length > 0 && <p className="alert error">Failed calls: {rec.failed_calls.join(', ')}</p>}
      <ol className="audit-docs">
        {rec.documents.map((d, i) => (
          <li key={i} className={item.kept.includes(i) ? 'kept' : ''}>
            <b>{d.title ?? `Document ${i + 1}`}</b>
            {item.kept.includes(i) && <span className="badge kept">kept</span>}
            <p>{d.content.length > 400 ? `${d.content.slice(0, 400)}…` : d.content}</p>
          </li>
        ))}
      </ol>
    </div>
  )
}

export function AuditPage({ replay = false, openId = null }: { replay?: boolean; openId?: string | null }) {
  const [data, setData] = useState<AuditList | null>(null)
  const [bundle, setBundle] = useState<ReplayAudit | null>(null)
  const [filter, setFilter] = useState<AuditOutcome | undefined>()
  const [open, setOpen] = useState<string | null>(openId)
  const [err, setErr] = useState<string | null>(null)

  const load = useCallback(() => {
    if (replay) {
      fetchReplayAudit().then(
        (b) => {
          setBundle(b)
          setData({ stats: b.stats, items: filter ? b.items.filter((i) => i.outcome === filter) : b.items })
          setErr(null)
        },
        (e: Error) => setErr(e.message),
      )
      return
    }
    fetchAudit(filter).then(
      (d) => {
        setData(d)
        setErr(null)
      },
      (e: Error) => setErr(e.message),
    )
  }, [filter, replay])
  useEffect(() => {
    load()
    if (replay) return
    const id = window.setInterval(load, 10000)
    return () => window.clearInterval(id)
  }, [load, replay])

  const s = data?.stats
  return (
    <main className="audit-page">
      <section className="card">
        <div className="card-title">
          <h2>Audit log{replay ? ' (recorded replay)' : ''}</h2>
          {!replay && <button className="btn small" onClick={load}>Refresh</button>}
        </div>
        {replay && (
          <div className="replay-banner" role="note">
            <b className="rec">RECORDED · NOT LIVE.</b> The three replay cases rendered as audit-log entries, from stored traces; no
            backend or model is called.
          </div>
        )}
        <p className="muted small">
          Requests sent to <code>POST /v1/chat/completions</code> with a non-standard <code>documents</code> list. For each, MUG records
          what plain RAG would have answered, whether the documents moved the answer, and whether a verifier model returned a supporting
          quote found verbatim in a kept passage. In <em>shadow</em> mode the client always receives the plain-RAG answer and the request is
          only labelled.
        </p>
        {err && <div className="alert error">{err}</div>}
        {s && (
          <div className="audit-stats seven">
            <button className={`stat ${filter === undefined ? 'on' : ''}`} onClick={() => setFilter(undefined)}>
              <b>{s.n}</b>
              <span>All requests</span>
            </button>
            {OUTCOMES.map((o) => (
              <button key={o.key} title={o.hint} className={`stat out-${o.key} ${filter === o.key ? 'on' : ''}`}
                onClick={() => setFilter(filter === o.key ? undefined : o.key)}>
                <b>{s.by_outcome[o.key] ?? 0}</b>
                <span>{o.label}</span>
              </button>
            ))}
          </div>
        )}
        {s && s.n > 0 && (
          <p className="muted small">
            Mean {Math.round(s.mean_prompt_tokens).toLocaleString()} prompt tokens · {(s.mean_wall_ms / 1000).toFixed(1)} s per request
          </p>
        )}
      </section>

      <section className="card">
        {data && data.items.length === 0 && <p className="muted">No audited requests yet.</p>}
        <table className="audit-table">
          <tbody>
            {data?.items.map((it) => (
              <Fragment key={it.id}>
                <tr className={open === it.id ? 'open' : ''} onClick={() => setOpen(open === it.id ? null : it.id)}>
                  <td className="small muted nowrap">{replay ? 'recorded' : new Date(it.ts * 1000).toLocaleString()}</td>
                  <td className="nowrap">
                    <span className={`badge out out-${it.outcome}`}>{OUTCOMES.find((o) => o.key === it.outcome)?.label ?? it.outcome}</span>
                    {it.proxy_mode === 'shadow' && <span className="badge shadow">shadow</span>}
                  </td>
                  <td className="q">{it.question}</td>
                  <td className="small muted nowrap">{it.n_docs} docs · {it.client}</td>
                </tr>
                {open === it.id && (
                  <tr className="detail-row">
                    <td colSpan={4}><Detail item={it} replay={bundle} /></td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  )
}
