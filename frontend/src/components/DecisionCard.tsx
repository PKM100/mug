import type { AskRequest, Candidate, ResultEvent, StartEvent } from '../api'
import { GATE_INFO, MODE_LABEL } from '../format'
import { AnswerText } from './AnswerText'
import { Gauge } from './Gauge'

interface Props {
  res: ResultEvent | null
  request: AskRequest | null
  start: StartEvent | null
  running: boolean
  stage: string
}

const DEFAULT_NOTE: Record<string, string> = {
  naive: 'Baseline: the full retrieved pool is always admitted.',
  closed_book: 'Baseline: no retrieval; answered from parametric knowledge.',
}

function McqAnswer({ letter, options, size = 'md' }: { letter: string | null; options: Record<string, string>; size?: 'md' | 'lg' }) {
  if (!letter) return <span className="muted">No answer</span>
  return (
    <span className={`mcq-answer ${size}`}>
      <span className="letter">{letter}</span>
      <span>{options[letter] ?? ''}</span>
    </span>
  )
}

function AnswerBox(props: {
  title: string
  sub: string
  value: string | null
  isMcq: boolean
  options: Record<string, string> | null
  docs: Candidate[]
  differs: boolean
  na: string
}) {
  const { title, sub, value, isMcq, options, docs, differs, na } = props
  return (
    <div className={`answer-box ${differs ? 'differs' : ''} ${value == null ? 'na' : ''}`}>
      <div className="answer-box-head">
        <strong>{title}</strong>
        <span className="muted small">{sub}</span>
      </div>
      {value == null ? (
        <span className="muted small">{na}</span>
      ) : isMcq && options ? (
        <McqAnswer letter={value} options={options} />
      ) : (
        <div className="open-answer scroll">
          <AnswerText text={value} docs={docs} />
        </div>
      )}
    </div>
  )
}

export function DecisionCard({ res, request, start, running, stage }: Props) {
  if (!res) {
    return (
      <section className="card decision pending">
        {running ? (
          <>
            <div className="verdict deciding">
              <span className="spinner" />
              <div>
                <div className="verdict-title">Deciding…</div>
                <div className="muted">{stage}</div>
              </div>
            </div>
            {start && (
              <p className="muted small">
                {GATE_INFO[start.gate]?.label ?? start.gate} gate · {start.question_type === 'mcq' ? 'multiple-choice' : 'open-ended'} · answer{' '}
                {start.answer_model} · {start.scout_model ? `scout ${start.scout_model}` : `probe ${start.probe_model}`}
                {start.logprobs ? ' · logprobs' : ''}
              </p>
            )}
          </>
        ) : (
          <div className="placeholder">
            <div className="placeholder-icon">⚖</div>
            <h3>Will the evidence change the answer?</h3>
            <p className="muted">
              MUG answers once <em>closed-book</em> and once <em>with retrieved evidence</em>. Evidence is admitted only when it moves
              the answer — relevance alone is not enough.
            </p>
          </div>
        )}
      </section>
    )
  }

  const r = res.result
  const isMcq = res.question_type === 'mcq'
  const options = res.options
  const gate = r.gate
  const tauM = request?.tau_m ?? null
  const tauSem = request?.tau_sem ?? 0.5
  const tauG = request?.tau_g ?? null

  let threshold: number | null = null
  let tLabel = 'τ'
  let hint = ''
  if (gate === 'flip') {
    if (isMcq) hint = 'binary: 1 = answer flipped'
    else {
      threshold = tauSem
      tLabel = 'τ_sem'
      hint = '1 − P(equivalent)'
    }
  } else if (gate === 'logit_shift' || gate === 'grounded') {
    threshold = isMcq ? tauM : tauSem
    tLabel = isMcq ? 'τ_m' : 'τ_sem'
    hint = isMcq ? 'total variation' : '1 − P(equivalent)'
  }

  const both = r.closed_book_answer != null && r.context_answer != null
  const differs = both && (isMcq ? r.closed_book_answer !== r.context_answer : (r.movement ?? 0) >= tauSem)
  const finalDocs = r.admitted ? r.kept_context : []
  const verdict = r.admitted ? 'ADMITTED' : 'WITHHELD'
  const note = r.note || DEFAULT_NOTE[r.mode] || ''

  return (
    <section className={`card decision ${r.admitted ? 'admitted' : 'withheld'}`}>
      <div className="verdict">
        <div className="verdict-icon" aria-hidden>
          {r.admitted ? '✓' : '⊘'}
        </div>
        <div className="verdict-main">
          <div className="verdict-title">
            Context <span>{verdict}</span>
          </div>
          <div className="verdict-sub">
            <span className="tag">{GATE_INFO[gate]?.label ?? gate}</span>
            <span className="tag mode">mode: {MODE_LABEL[r.mode] ?? r.mode}</span>
            <span className="tag">{isMcq ? 'multiple-choice' : 'open-ended'}</span>
          </div>
          {note && <p className="note">{note}</p>}
        </div>
      </div>

      {(r.movement != null || r.groundedness != null) && (
        <div className="gauges">
          {r.movement != null && <Gauge label="Answer movement" value={r.movement} threshold={threshold} thresholdLabel={tLabel} hint={hint} />}
          {r.groundedness != null && (
            <Gauge label="Groundedness" value={r.groundedness} threshold={tauG} thresholdLabel="τ_g" hint="P(evidence entails answer)" tone="ground" />
          )}
        </div>
      )}

      <div className="answers-compare">
        <AnswerBox
          title="Closed-book"
          sub="no evidence"
          value={r.closed_book_answer}
          isMcq={isMcq}
          options={options}
          docs={[]}
          differs={differs}
          na={gate === 'naive' ? 'Not probed by the naive gate' : 'Not available'}
        />
        <div className={`vs ${differs ? 'differs' : ''}`} title={differs ? 'Answers differ' : 'Answers agree'}>
          {both ? (differs ? '≠' : '=') : '·'}
        </div>
        <AnswerBox
          title="With context"
          sub="full retrieved pool"
          value={r.context_answer}
          isMcq={isMcq}
          options={options}
          docs={res.pool}
          differs={differs}
          na={gate === 'closed_book' ? 'Closed-book gate never retrieves' : 'Not available'}
        />
      </div>

      <div className="final-answer">
        <div className="final-label">Final answer</div>
        {r.answer == null && !r.answer_text ? (
          <span className="muted">The model did not return an answer.</span>
        ) : isMcq && options ? (
          <McqAnswer letter={r.answer} options={options} size="lg" />
        ) : (
          <AnswerText text={r.answer_text} docs={finalDocs} />
        )}
        {r.admitted && finalDocs.length > 0 && (
          <p className="muted small">
            Answered on {finalDocs.length} admitted passage{finalDocs.length === 1 ? '' : 's'}
            {!isMcq && ' — click a [Doc n] citation to jump to it.'}
          </p>
        )}
        {!r.admitted && gate !== 'naive' && (
          <p className="withheld-hint" role="note">
            <b>Withheld is not endorsed.</b> This is the model&rsquo;s unaided (closed-book) answer. MUG did not check it against the
            literature; it only found that the retrieved evidence did not {r.mode === 'closed_book' || r.mode === 'no_evidence' ? 'change it' : 'support a change'}.
          </p>
        )}
      </div>

      {r.verification && (
        <div className={`verification ${r.verification.supported ? 'ok' : 'no'}`}>
          <div className="verification-head">
            <strong>Evidence verifier</strong>
            {r.verification.model && <span className="tag">{r.verification.model}</span>}
            <span className={`badge ${r.verification.supported ? 'kept' : 'withheld'}`}>
              {r.verification.supported ? 'Quote found + model judged supportive' : 'No verifiable quote'}
            </span>
          </div>
          {r.verification.supported && r.verification.quote ? (
            <blockquote className="audit-quote">
              &ldquo;{r.verification.quote}&rdquo;{r.verification.doc != null && <span className="muted small"> — kept Doc {r.verification.doc}</span>}
            </blockquote>
          ) : (
            <p className="muted small">
              The verifier returned no quote that occurs verbatim in a kept passage{r.verification.error ? ' (verifier call failed)' : ''}, so the
              moved answer was not admitted.
            </p>
          )}
          <p className="muted small">
            An LLM judgement with a verbatim string-match check on the quote; it is not a check of clinical correctness.
          </p>
        </div>
      )}
    </section>
  )
}
