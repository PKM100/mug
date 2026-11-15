import { useState } from 'react'
import { supportsLogprobs, type AppConfig, type GateName } from '../api'
import { GATE_INFO } from '../format'
import type { SettingsState } from '../state'

interface Props {
  config: AppConfig | null
  settings: SettingsState
  onChange: (s: SettingsState) => void
  isMcq: boolean
  disabled: boolean
}

function Slider(props: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  hint?: string
  disabled?: boolean
}) {
  const { label, value, min, max, step, onChange, hint, disabled } = props
  return (
    <label className={`slider ${disabled ? 'is-disabled' : ''}`}>
      <span className="slider-head">
        <span>{label}</span>
        <output>{step < 1 ? value.toFixed(2) : value}</output>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <span className="muted small">{hint}</span>}
    </label>
  )
}

function Check(props: { label: string; checked: boolean; onChange: (v: boolean) => void; hint: string; disabled?: boolean }) {
  return (
    <label className={`toggle ${props.disabled ? 'is-disabled' : ''}`} title={props.hint}>
      <input type="checkbox" checked={props.checked} onChange={(e) => props.onChange(e.target.checked)} />
      <span className="switch" />
      <span>
        {props.label}
        <span className="muted small block">{props.hint}</span>
      </span>
    </label>
  )
}

export function SettingsPanel({ config, settings: s, onChange, isMcq, disabled }: Props) {
  const [open, setOpen] = useState(false)
  const set = (patch: Partial<SettingsState>) => onChange({ ...s, ...patch })
  const gates: GateName[] = config?.gates ?? ['flip', 'grounded', 'logit_shift', 'naive', 'closed_book']
  const models = config?.models ?? [s.answer_model]
  const lpModels = config?.logprob_models ?? [s.probe_model]
  const needsLogprobs = s.gate === 'logit_shift' || s.gate === 'grounded'
  const noLogprobs = needsLogprobs && isMcq && !supportsLogprobs(s.answer_model, lpModels)
  const retrieves = s.gate !== 'closed_book'
  const gated = retrieves && s.gate !== 'naive'
  const cascade = gated && !!s.scout_model

  return (
    <section className="card settings">
      <button type="button" className="collapse-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        <h2>Settings</h2>
        <span className="muted small">
          {GATE_INFO[s.gate]?.label ?? s.gate} · {s.answer_model}
          {cascade ? ` · scout ${s.scout_model}` : ''}
        </span>
        <span className={`chev ${open ? 'open' : ''}`} aria-hidden>
          ▾
        </span>
      </button>

      <fieldset className="gate-list" disabled={disabled}>
        <legend className="sr-only">Gate</legend>
        {gates.map((g) => (
          <label key={g} className={`gate-option ${s.gate === g ? 'selected' : ''}`}>
            <input type="radio" name="gate" value={g} checked={s.gate === g} onChange={() => set({ gate: g })} />
            <span>
              <strong>{GATE_INFO[g]?.label ?? g}</strong>
              <span className="muted small block">{GATE_INFO[g]?.blurb}</span>
            </span>
          </label>
        ))}
      </fieldset>
      {noLogprobs && (
        <p className="field-hint warn">
          ⚠ <strong>{s.answer_model}</strong> does not expose token logprobs; {GATE_INFO[s.gate].label} will fall back to
          binary (flip) movement. Use one of: {lpModels.join(', ')}.
        </p>
      )}

      {open && (
        <fieldset className="settings-body" disabled={disabled}>
          <div className="grid-2">
            <label className="field">
              <span>Answer model</span>
              <select value={s.answer_model} onChange={(e) => set({ answer_model: e.target.value })}>
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                    {supportsLogprobs(m, lpModels) ? ' · logprobs' : ''}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Probe model</span>
              <select value={s.probe_model} onChange={(e) => set({ probe_model: e.target.value })}>
                {lpModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <span className="muted small">Equivalence &amp; groundedness probes (needs logprobs)</span>
            </label>
          </div>

          <label className={`field ${gated ? '' : 'is-disabled'}`}>
            <span>Scout model (cascade)</span>
            <select value={s.scout_model} onChange={(e) => set({ scout_model: e.target.value })} disabled={!gated}>
              <option value="">Off — answer model runs its own gate</option>
              {models
                .filter((m) => m !== s.answer_model)
                .map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
            </select>
            <span className="muted small">
              A cheaper model runs the probes and filter; the answer model reads only the admitted evidence. In our tests,
              gpt-4o-mini scouting for gpt-4.1 matched gpt-4.1 naive RAG at ~17% of its cost (cheap fixed policies did too).
            </span>
          </label>

          <label className={`field ${s.gate === 'flip' ? '' : 'is-disabled'}`}>
            <span>Evidence verifier</span>
            <select value={s.verifier_model} onChange={(e) => set({ verifier_model: e.target.value })} disabled={s.gate !== 'flip'}>
              <option value="">Off</option>
              {models
                .filter((m) => m.startsWith('claude') || m.startsWith('gemini'))
                .map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
            </select>
            <span className="muted small">
              Flip gate only. After evidence moves the answer, an independent model must quote a passage that directly
              establishes it; otherwise the context is withheld. Pick a different model family from the answer model.
            </span>
          </label>

          <div className="grid-2">
            <Slider label="Textbook passages (k)" value={s.k_textbook} min={0} max={12} step={1} onChange={(v) => set({ k_textbook: v })} disabled={!retrieves} />
            <Slider label="PubMed abstracts (k)" value={s.k_pubmed} min={0} max={10} step={1} onChange={(v) => set({ k_pubmed: v })} disabled={!retrieves} />
          </div>

          <div className="toggles">
            <Check label="Rewrite PubMed query" checked={s.rewrite_query} onChange={(v) => set({ rewrite_query: v })} hint="LLM turns the question into a PubMed search term" disabled={!retrieves} />
            <Check label="Overlap retrieval" checked={s.overlap_retrieval} onChange={(v) => set({ overlap_retrieval: v })} hint="Launch the closed-book probe while retrieval runs" disabled={!retrieves} />
            <Check label="Speculative filter" checked={s.speculative_filter} onChange={(v) => set({ speculative_filter: v })} hint="Flip gate: run the evaluator in parallel; discard if no flip" disabled={s.gate !== 'flip'} />
          </div>

          <div className="grid-3">
            <Slider label="τ_m movement" value={s.tau_m} min={0} max={1} step={0.01} onChange={(v) => set({ tau_m: v })} hint="MCQ logit-shift / grounded" />
            <Slider label="τ_g groundedness" value={s.tau_g} min={0} max={1} step={0.01} onChange={(v) => set({ tau_g: v })} hint="Grounded gate" />
            <Slider label="τ_sem semantic" value={s.tau_sem} min={0} max={1} step={0.01} onChange={(v) => set({ tau_sem: v })} hint="Open-ended movement" />
          </div>
        </fieldset>
      )}
    </section>
  )
}
