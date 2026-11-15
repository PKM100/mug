import { EXAMPLES, type Example } from '../examples'
import { LETTERS } from '../format'
import type { FormState } from '../state'

interface Props {
  form: FormState
  onChange: (f: FormState) => void
  onSubmit: () => void
  onCompare: () => void
  onCancel: () => void
  running: boolean
  comparing: boolean
}

const MAX_PASSAGES = 5

export function Composer({ form, onChange, onSubmit, onCompare, onCancel, running, comparing }: Props) {
  const set = (patch: Partial<FormState>) => onChange({ ...form, ...patch })
  const busy = running || comparing
  const filledOptions = form.options.filter((o) => o.trim()).length
  const mcqInvalid = form.mcq && filledOptions < 2
  const canSubmit = form.question.trim().length >= 3 && !mcqInvalid && !busy

  const loadExample = (ex: Example) =>
    onChange({
      question: ex.question,
      mcq: !!ex.options,
      options: ex.options ? [...ex.options] : ['', '', '', ''],
      passages: form.passages,
    })

  return (
    <section className="card composer">
      <div className="card-title">
        <h2>Ask a question</h2>
      </div>
      <div className="examples" aria-label="Example questions">
        {EXAMPLES.map((ex) => (
          <button key={ex.label} type="button" className="chip" onClick={() => loadExample(ex)} disabled={busy} title={ex.question}>
            <span className={`chip-kind k-${ex.kind === 'Open' ? 'open' : ex.kind === 'MCQ' ? 'mcq' : 'ynm'}`}>{ex.kind}</span>
            {ex.label}
          </button>
        ))}
      </div>

      <textarea
        className="question-input"
        placeholder="e.g. A 26-year-old woman presents with fever, flank pain and WBC casts… or an open question like “Does metformin reduce colorectal cancer risk?”"
        value={form.question}
        rows={5}
        maxLength={4000}
        onChange={(e) => set({ question: e.target.value })}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && canSubmit) onSubmit()
        }}
      />

      <label className="toggle">
        <input type="checkbox" checked={form.mcq} onChange={(e) => set({ mcq: e.target.checked })} />
        <span className="switch" />
        Multiple choice
        <span className="muted small">{form.mcq ? 'answer is an option letter' : 'open-ended (semantic movement)'}</span>
      </label>

      {form.mcq && (
        <div className="options">
          {form.options.map((opt, i) => (
            <div className="option-row" key={i}>
              <span className="option-letter">{LETTERS[i]}</span>
              <input
                value={opt}
                placeholder={`Option ${LETTERS[i]}`}
                onChange={(e) => {
                  const options = [...form.options]
                  options[i] = e.target.value
                  set({ options })
                }}
              />
              <button
                type="button"
                className="icon-btn"
                aria-label={`Remove option ${LETTERS[i]}`}
                disabled={form.options.length <= 2}
                onClick={() => set({ options: form.options.filter((_, j) => j !== i) })}
              >
                ×
              </button>
            </div>
          ))}
          {form.options.length < LETTERS.length && (
            <button type="button" className="link-btn" onClick={() => set({ options: [...form.options, ''] })}>
              + Add option
            </button>
          )}
          {mcqInvalid && <p className="field-hint warn">Provide at least two options.</p>}
        </div>
      )}

      <div className="passages">
        {form.passages.map((p, i) => (
          <div className="passage-row" key={i}>
            <div className="passage-head">
              <span className="badge user">User passage {i + 1}</span>
              <button type="button" className="icon-btn" aria-label="Remove passage" onClick={() => set({ passages: form.passages.filter((_, j) => j !== i) })}>
                ×
              </button>
            </div>
            <textarea
              rows={3}
              value={p}
              placeholder="Paste an abstract, guideline excerpt, or note…"
              onChange={(e) => {
                const passages = [...form.passages]
                passages[i] = e.target.value
                set({ passages })
              }}
            />
          </div>
        ))}
        {form.passages.length < MAX_PASSAGES && (
          <button type="button" className="link-btn" onClick={() => set({ passages: [...form.passages, ''] })}>
            + Add your own passage {form.passages.length > 0 && `(${form.passages.length}/${MAX_PASSAGES})`}
          </button>
        )}
      </div>

      <div className="actions">
        {running ? (
          <button type="button" className="btn danger" onClick={onCancel}>
            Stop
          </button>
        ) : (
          <button type="button" className="btn primary" onClick={onSubmit} disabled={!canSubmit}>
            Ask MUG
          </button>
        )}
        <button type="button" className="btn" onClick={onCompare} disabled={!canSubmit} title="Run the same question through all 5 gates">
          Compare gates
        </button>
        <span className="muted small kbd-hint">⌘/Ctrl + Enter</span>
      </div>
    </section>
  )
}
