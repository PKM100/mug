import { useEffect, useState } from 'react'
import { fetchReplayCases, type ReplayBundle } from '../api'
import { CostStrip } from './CostStrip'
import { DecisionCard } from './DecisionCard'
import { EvidencePanel } from './EvidencePanel'
import { Timeline } from './Timeline'

/** Deterministic replay of recorded cases: renders stored traces with the live components, no backend or model calls. */
export function ReplayPage({ caseKey }: { caseKey: string | null }) {
  const [bundle, setBundle] = useState<ReplayBundle | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    fetchReplayCases().then(setBundle, (e: Error) => setErr(e.message))
  }, [])

  if (err) return <main className="audit-page"><div className="alert error">Could not load the replay bundle: {err}</div></main>
  if (!bundle) return <main className="audit-page"><p className="muted">Loading recorded cases…</p></main>
  const c = bundle.cases.find((x) => x.key === caseKey) ?? bundle.cases[0]
  const res = c.result
  const p = c.provenance

  return (
    <main className="layout">
      <aside className="left">
        <section className="card">
          <div className="card-title">
            <h2>Replay cases</h2>
          </div>
          <nav className="replay-cases">
            {bundle.cases.map((x, i) => (
              <a key={x.key} href={`#/replay/${x.key}`} className={x.key === c.key ? 'on' : ''}>
                <b>
                  {String.fromCharCode(97 + i)}. {x.title}
                </b>
                <small>{x.provenance.qid}</small>
              </a>
            ))}
            <a href="#/replay/audit">
              <b>Audit page for these cases</b>
              <small>recorded audit-log entries</small>
            </a>
          </nav>
        </section>
      </aside>
      <div className="right">
        <div className="replay-banner" role="note">
          <b className="rec">RECORDED · NOT LIVE.</b> {bundle.note} {c.blurb}
          <ul className="small">
            <li>
              Source: <code>{p.run}</code>, question <code>{p.qid}</code> (gold answer {p.gold}); recorded {p.recorded}.
            </li>
            {p.verifier && (
              <li>
                Verifier verdict: <code>{p.verifier}</code>.
              </li>
            )}
            {p.differences.map((d) => (
              <li key={d}>{d}</li>
            ))}
          </ul>
        </div>
        <div className="question-echo">
          <span className="eyebrow">Question</span>
          <p>{res.question}</p>
          {res.options && (
            <ol className="echo-options">
              {Object.entries(res.options).map(([k, v]) => (
                <li key={k} className={res.result.answer === k ? 'chosen' : ''}>
                  <b>{k}.</b> {v}
                </li>
              ))}
            </ol>
          )}
        </div>
        <DecisionCard res={res} request={c.request} start={null} running={false} stage="" />
        <CostStrip result={res.result} summary={res.summary} spans={res.spans} />
        <Timeline spans={res.spans} running={false} elapsedMs={0} wallMs={res.summary.wall_ms} />
        <EvidencePanel pool={res.pool} poolTokens={res.result.pool_tokens} result={res.result} running={false} closedBookGate={false} />
      </div>
    </main>
  )
}
