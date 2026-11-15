// Types mirror backend/app.py, backend/mug/pipeline.py, gates.py and trace.py.

export type GateName = 'flip' | 'grounded' | 'logit_shift' | 'naive' | 'closed_book'
export type QuestionType = 'mcq' | 'open'
export type SpanKind = 'llm' | 'retrieval' | 'decision'
export type Source = 'textbook' | 'pubmed' | 'user'

export interface Defaults {
  gate: GateName
  answer_model: string
  probe_model: string
  k_textbook: number
  k_pubmed: number
  tau_m: number
  tau_g: number
  tau_sem: number
}

export interface AppConfig {
  gates: GateName[]
  models: string[]
  logprob_models: string[]
  defaults: Defaults
}

export interface Health {
  ok: boolean
  textbooks_ready: boolean
  textbooks_error: string | null
  textbook_snippets: number
}

export interface AskRequest {
  question: string
  options: Record<string, string> | null
  gate: GateName
  answer_model: string | null
  probe_model: string | null
  k_textbook: number
  k_pubmed: number
  rewrite_query: boolean
  user_passages: string[]
  speculative_filter: boolean
  overlap_retrieval: boolean
  /** Cascade: a cheap model runs the probes and filter; the answer model only sees admitted context. */
  scout_model?: string | null
  /** Independent model that must quote a passage establishing the moved answer before flip admits it. */
  verifier_model?: string | null
  tau_m: number | null
  tau_g: number | null
  tau_sem: number
}

export interface Span {
  name: string
  kind: SpanKind
  start_ms: number
  end_ms: number
  duration_ms: number
  model: string | null
  prompt_tokens: number
  completion_tokens: number
  speculative: boolean
  used: boolean
  detail: Record<string, unknown>
}

export interface Candidate {
  id: string
  source: Source
  title: string
  content: string
  url: string | null
  score?: number
  meta?: {
    journal?: string
    year?: string
    pub_types?: string[]
    quality?: { level: number; label: string; retracted: boolean; age_years: number | null; flags: string[] }
  }
  pool_index?: number
  /** Present when the evaluator compressed the passage to its decisive sentence (content). */
  full_content?: string
}

export interface GateResult {
  gate: GateName
  mode: string
  answer: string | null
  answer_text: string
  admitted: boolean
  kept_context: Candidate[]
  closed_book_answer: string | null
  context_answer: string | null
  movement: number | null
  groundedness: number | null
  retained_tokens: number
  pool_tokens: number
  note: string
  /** Present when the flip gate ran an evidence verifier on a moved answer. */
  verification?: Verification | null
}

/** `supported` means: the verifier model judged a kept passage supportive AND its quote occurs verbatim in it. */
export interface Verification {
  supported: boolean
  doc: number | null
  quote: string
  model?: string
  error?: boolean
}

export interface Summary {
  wall_ms: number
  llm_calls: number
  llm_calls_used: number
  prompt_tokens: number
  completion_tokens: number
  wasted_prompt_tokens: number
  sequential_ms: number
}

export interface StartEvent {
  type: 'start'
  question_type: QuestionType
  gate: GateName
  answer_model: string
  probe_model: string
  scout_model?: string | null
  verifier_model?: string | null
  logprobs: boolean
}
export interface SpanEvent {
  type: 'span'
  span: Span
}
export interface PoolEvent {
  type: 'pool'
  pool: Candidate[]
  pool_tokens: number
}
export interface ResultEvent {
  type: 'result'
  question: string
  options: Record<string, string> | null
  question_type: QuestionType
  result: GateResult
  pool: Candidate[]
  summary: Summary
  spans: Span[]
}
export interface ErrorEvent {
  type: 'error'
  message: string
}
export type StreamEvent = StartEvent | SpanEvent | PoolEvent | ResultEvent | ErrorEvent

async function errorMessage(res: Response): Promise<string> {
  const text = await res.text().catch(() => '')
  try {
    const body = JSON.parse(text) as { detail?: unknown }
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) {
      return body.detail
        .map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).slice(1).join('.')}: ${d.msg ?? ''}`)
        .join('; ')
    }
  } catch {
    /* not JSON */
  }
  return `HTTP ${res.status}${text ? ` — ${text.slice(0, 200)}` : ''}`
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(await errorMessage(res))
  return (await res.json()) as T
}

export const fetchConfig = (): Promise<AppConfig> => getJSON<AppConfig>('/api/config')
export const fetchHealth = (): Promise<Health> => getJSON<Health>('/api/health')

export async function ask(req: AskRequest, signal?: AbortSignal): Promise<ResultEvent> {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })
  if (!res.ok) throw new Error(await errorMessage(res))
  return (await res.json()) as ResultEvent
}

/** POST + Server-Sent Events: parse `data: <json>` frames off a ReadableStream. */
export async function askStream(
  req: AskRequest,
  onEvent: (ev: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(req),
    signal,
  })
  if (!res.ok) throw new Error(await errorMessage(res))
  if (!res.body) throw new Error('Streaming not supported by this browser')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  const flush = (block: string) => {
    const data = block
      .split(/\r?\n/)
      .filter((l) => l.startsWith('data:'))
      .map((l) => l.slice(5).replace(/^ /, ''))
      .join('\n')
    if (!data) return
    try {
      onEvent(JSON.parse(data) as StreamEvent)
    } catch {
      /* ignore malformed frame */
    }
  }
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buf.search(/\r?\n\r?\n/)) >= 0) {
      const block = buf.slice(0, idx)
      buf = buf.slice(idx).replace(/^\r?\n\r?\n/, '')
      flush(block)
    }
  }
  buf += decoder.decode()
  if (buf.trim()) flush(buf)
}

export function supportsLogprobs(model: string, logprobModels: string[]): boolean {
  return logprobModels.some((m) => model === m || model.startsWith(m + '-'))
}

// ---------------------------------------------------------------- audit-proxy
export type AuditOutcome = 'no_effect' | 'supported' | 'unverified' | 'admitted' | 'no_documents' | 'failed'
export type ProxyMode = 'gate' | 'shadow'

export interface AuditItem {
  id: string
  ts: number
  client: string
  model: string
  verifier: string | null
  question: string
  n_docs: number
  outcome: AuditOutcome
  /** Text returned to the client: the gated answer in gate mode, the plain-RAG answer in shadow mode. */
  answer: string
  gate_answer?: string | null
  proxy_mode?: ProxyMode
  rag_answer: string | null
  closed_book_answer: string | null
  kept: number[]
  quote: string
  movement: number | null
  prompt_tokens: number
  wall_ms: number
}

export interface AuditList {
  stats: { n: number; by_outcome: Record<AuditOutcome, number>; mean_prompt_tokens: number; mean_wall_ms: number }
  items: AuditItem[]
}

export interface AuditRecord {
  result: GateResult
  documents: { content: string; title: string | null; url: string | null }[]
  summary: Summary
  proxy_mode?: ProxyMode
  failed_calls?: string[]
}

export const fetchAudit = (outcome?: AuditOutcome): Promise<AuditList> =>
  getJSON<AuditList>(`/api/audit?limit=200${outcome ? `&outcome=${outcome}` : ''}`)
export const fetchAuditRecord = (id: string): Promise<AuditRecord> => getJSON<AuditRecord>(`/api/audit/${id}`)

// ---------------------------------------------------------------- replay (recorded cases, no backend needed)
export interface ReplayCase {
  key: string
  title: string
  blurb: string
  request: AskRequest
  result: ResultEvent
  provenance: { run: string; qid: string; gold: string; recorded: string; verifier: string | null; differences: string[] }
}
export interface ReplayBundle {
  note: string
  cases: ReplayCase[]
}
export interface ReplayAudit extends AuditList {
  records: Record<string, AuditRecord>
}
const REPLAY_BASE = `${import.meta.env.BASE_URL}replay/`
export const fetchReplayCases = (): Promise<ReplayBundle> => getJSON<ReplayBundle>(`${REPLAY_BASE}cases.json`)
export const fetchReplayAudit = (): Promise<ReplayAudit> => getJSON<ReplayAudit>(`${REPLAY_BASE}audit.json`)
