import type { AppConfig, AskRequest, GateName } from './api'
import { LETTERS } from './format'

export interface FormState {
  question: string
  mcq: boolean
  options: string[]
  passages: string[]
}

export interface SettingsState {
  gate: GateName
  answer_model: string
  probe_model: string
  /** '' = no cascade */
  scout_model: string
  /** '' = no verifier */
  verifier_model: string
  k_textbook: number
  k_pubmed: number
  rewrite_query: boolean
  speculative_filter: boolean
  overlap_retrieval: boolean
  tau_m: number
  tau_g: number
  tau_sem: number
}

export const EMPTY_FORM: FormState = { question: '', mcq: false, options: ['', '', '', ''], passages: [] }

export const FALLBACK_SETTINGS: SettingsState = {
  gate: 'flip',
  answer_model: 'gpt-4o-mini',
  probe_model: 'gpt-4o-mini',
  scout_model: '',
  verifier_model: '',
  k_textbook: 4,
  k_pubmed: 3,
  rewrite_query: true,
  speculative_filter: false,
  overlap_retrieval: true,
  tau_m: 0.15,
  tau_g: 0.5,
  tau_sem: 0.5,
}

export function settingsFromConfig(cfg: AppConfig): SettingsState {
  const d = cfg.defaults
  return {
    ...FALLBACK_SETTINGS,
    gate: d.gate,
    answer_model: d.answer_model,
    probe_model: d.probe_model,
    k_textbook: d.k_textbook,
    k_pubmed: d.k_pubmed,
    tau_m: d.tau_m,
    tau_g: d.tau_g,
    tau_sem: d.tau_sem,
  }
}

export function buildRequest(form: FormState, s: SettingsState): AskRequest {
  let options: Record<string, string> | null = null
  if (form.mcq) {
    const filled = form.options.map((o) => o.trim()).filter(Boolean)
    const entries = filled.map((o, i) => [LETTERS[i], o] as const)
    options = entries.length ? Object.fromEntries(entries) : null
  }
  return {
    question: form.question.trim(),
    options,
    gate: s.gate,
    answer_model: s.answer_model,
    probe_model: s.probe_model,
    scout_model: s.scout_model || null,
    verifier_model: s.gate === 'flip' ? s.verifier_model || null : null,
    k_textbook: s.k_textbook,
    k_pubmed: s.k_pubmed,
    rewrite_query: s.rewrite_query,
    user_passages: form.passages.map((p) => p.trim()).filter(Boolean),
    speculative_filter: s.speculative_filter,
    overlap_retrieval: s.overlap_retrieval,
    tau_m: s.tau_m,
    tau_g: s.tau_g,
    tau_sem: s.tau_sem,
  }
}

export function formFromRequest(req: AskRequest): FormState {
  const opts = req.options ? Object.values(req.options) : []
  return {
    question: req.question,
    mcq: opts.length > 0,
    options: opts.length ? opts : ['', '', '', ''],
    passages: [...req.user_passages],
  }
}
