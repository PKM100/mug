import type { GateName, ResultEvent } from '../api'
import { GATE_INFO, MODE_LABEL, fmtInt, fmtMs, fmtNum, truncate } from '../format'

export interface CompareRow {
  gate: GateName
  status: 'pending' | 'running' | 'done' | 'error'
  res?: ResultEvent
  error?: string
}

interface Props {
  rows: CompareRow[]
  running: boolean
  onCancel: () => void
  onView: (res: ResultEvent) => void
  onClose: () => void
}

export function CompareGates({ rows, running, onCancel, onView, onClose }: Props) {
  const done = rows.filter((r) => r.status === 'done' || r.status === 'error').length
  const current = rows.find((r) => r.status === 'running')
  const answers = new Set(rows.filter((r) => r.res?.result.answer_text).map((r) => r.res!.result.answer_text))
  const isMcq = rows.find((r) => r.res)?.res?.question_type === 'mcq'

  return (
    <section className="card compare">
      <div className="card-title">
        <h2>Gate comparison</h2>
        <span className="muted small">
          {running ? `Running ${GATE_INFO[current?.gate ?? '']?.label ?? '…'} (${done + 1}/${rows.length})` : `${done}/${rows.length} complete`}
          {!running && isMcq && answers.size > 0 && ` · ${answers.size === 1 ? 'all gates agree' : `${answers.size} distinct answers`}`}
        </span>
        <div className="spacer" />
        {running ? (
          <button type="button" className="btn small danger" onClick={onCancel}>
            Stop
          </button>
        ) : (
          <button type="button" className="btn small" onClick={onClose}>
            Close
          </button>
        )}
      </div>
      <div className="progress" aria-hidden>
        <span style={{ width: `${(done / rows.length) * 100}%` }} />
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Gate</th>
              <th>Mode</th>
              <th>Answer</th>
              <th>Admitted</th>
              <th className="num">Movement</th>
              <th className="num">Grounded</th>
              <th className="num">Retained tok</th>
              <th className="num">Calls</th>
              <th className="num">Wall</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const r = row.res?.result
              const s = row.res?.summary
              return (
                <tr key={row.gate} className={row.status}>
                  <td>
                    <strong>{GATE_INFO[row.gate]?.label ?? row.gate}</strong>
                  </td>
                  {row.status === 'done' && r && s ? (
                    <>
                      <td>{MODE_LABEL[r.mode] ?? r.mode}</td>
                      <td className="ans" title={r.answer_text}>
                        {truncate(r.answer_text || '—', isMcq ? 60 : 90)}
                      </td>
                      <td>
                        <span className={`yn ${r.admitted ? 'yes' : 'no'}`}>{r.admitted ? 'admitted' : 'withheld'}</span>
                      </td>
                      <td className="num">{fmtNum(r.movement)}</td>
                      <td className="num">{fmtNum(r.groundedness)}</td>
                      <td className="num">
                        {fmtInt(r.retained_tokens)}
                        {r.pool_tokens > 0 && <span className="muted"> / {fmtInt(r.pool_tokens)}</span>}
                      </td>
                      <td className="num">{s.llm_calls}</td>
                      <td className="num">{fmtMs(s.wall_ms)}</td>
                      <td>
                        <button type="button" className="link-btn small" onClick={() => onView(row.res!)}>
                          View
                        </button>
                      </td>
                    </>
                  ) : (
                    <td colSpan={9} className={`muted ${row.status === 'error' ? 'err' : ''}`}>
                      {row.status === 'running' ? (
                        <>
                          <span className="spinner sm" /> running…
                        </>
                      ) : row.status === 'error' ? (
                        row.error
                      ) : (
                        'queued'
                      )}
                    </td>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
