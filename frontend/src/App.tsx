import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ask,
  askStream,
  fetchConfig,
  fetchHealth,
  type AppConfig,
  type AskRequest,
  type Candidate,
  type GateName,
  type Health,
  type ResultEvent,
  type Span,
  type StartEvent,
} from './api'
import { AuditPage } from './components/AuditPage'
import { CompareGates, type CompareRow } from './components/CompareGates'
import { Composer } from './components/Composer'
import { CostStrip } from './components/CostStrip'
import { DecisionCard } from './components/DecisionCard'
import { EvidencePanel } from './components/EvidencePanel'
import { Header, type PubmedStatus } from './components/Header'
import { HistoryList } from './components/HistoryList'
import { ReplayPage } from './components/ReplayPage'
import { SettingsPanel } from './components/SettingsPanel'
import { Timeline } from './components/Timeline'
import { loadHistory, saveHistory, type HistoryEntry } from './history'
import {
  EMPTY_FORM,
  FALLBACK_SETTINGS,
  buildRequest,
  formFromRequest,
  settingsFromConfig,
  type FormState,
  type SettingsState,
} from './state'

type Status = 'idle' | 'running' | 'done' | 'error'

interface RunState {
  status: Status
  request: AskRequest | null
  start: StartEvent | null
  spans: Span[]
  pool: Candidate[]
  poolTokens: number | null
  result: ResultEvent | null
  error: string | null
  t0: number
}

const IDLE: RunState = { status: 'idle', request: null, start: null, spans: [], pool: [], poolTokens: null, result: null, error: null, t0: 0 }
const ALL_GATES: GateName[] = ['flip', 'grounded', 'logit_shift', 'naive', 'closed_book']

function stageText(run: RunState): string {
  if (run.request?.gate === 'closed_book') return 'Answering closed-book…'
  if (run.poolTokens == null) return 'Retrieving evidence (textbooks + PubMed) while probing closed-book…'
  const names = run.spans.map((s) => s.name)
  if (names.some((n) => n.startsWith('decide'))) return 'Evidence moved the answer — filtering and re-answering…'
  if (names.includes('probe(full pool)')) return 'Measuring answer movement…'
  return 'Comparing closed-book vs. with-context answers…'
}

function pubmedFromSpans(spans: Span[]): { status: PubmedStatus; note?: string } | null {
  const s = [...spans].reverse().find((x) => x.name === 'retrieve(pubmed)')
  if (!s) return null
  const err = s.detail?.error
  if (typeof err === 'string') return { status: 'error', note: err }
  return { status: 'ok', note: `Last query: ${String(s.detail?.query ?? '')} · ${String(s.detail?.n ?? 0)} hits` }
}

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [route, setRoute] = useState(window.location.hash)
  useEffect(() => {
    const on = () => setRoute(window.location.hash)
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [pubmed, setPubmed] = useState<{ status: PubmedStatus; note?: string }>({ status: 'unknown' })
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [settings, setSettings] = useState<SettingsState>(FALLBACK_SETTINGS)
  const [run, setRun] = useState<RunState>(IDLE)
  const [now, setNow] = useState<number>(() => performance.now())
  const [compare, setCompare] = useState<CompareRow[] | null>(null)
  const [comparing, setComparing] = useState(false)
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory())
  const [activeId, setActiveId] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const compareAbortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    fetchConfig()
      .then((cfg) => {
        setConfig(cfg)
        setSettings(settingsFromConfig(cfg))
      })
      .catch((e: unknown) => setHealthError(errText(e)))
  }, [])

  const ready = health?.textbooks_ready ?? false
  useEffect(() => {
    let alive = true
    const tick = () =>
      fetchHealth()
        .then((h) => {
          if (!alive) return
          setHealth(h)
          setHealthError(null)
        })
        .catch((e: unknown) => {
          if (alive) setHealthError(errText(e))
        })
    void tick()
    const id = window.setInterval(tick, ready ? 60_000 : 8_000)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [ready])

  useEffect(() => {
    if (run.status !== 'running') return
    const id = window.setInterval(() => setNow(performance.now()), 100)
    return () => window.clearInterval(id)
  }, [run.status])

  const addHistory = useCallback((request: AskRequest, result: ResultEvent): string => {
    const entry: HistoryEntry = { id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, ts: Date.now(), request, result }
    setHistory((h) => saveHistory([entry, ...h]))
    return entry.id
  }, [])

  const submit = useCallback(async () => {
    const request = buildRequest(form, settings)
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    const t0 = performance.now()
    setNow(t0)
    setActiveId(null)
    setRun({ ...IDLE, status: 'running', request, t0 })
    if (window.matchMedia('(max-width: 1020px)').matches) {
      requestAnimationFrame(() => document.querySelector('.right')?.scrollIntoView({ behavior: 'smooth' }))
    }
    let final: ResultEvent | null = null
    let streamError = false
    try {
      await askStream(
        request,
        (ev) => {
          switch (ev.type) {
            case 'start':
              setRun((r) => ({ ...r, start: ev }))
              break
            case 'span':
              setRun((r) => ({ ...r, spans: [...r.spans, ev.span] }))
              break
            case 'pool':
              setRun((r) => ({ ...r, pool: ev.pool, poolTokens: ev.pool_tokens }))
              break
            case 'result':
              final = ev
              setRun((r) => ({ ...r, status: 'done', result: ev, spans: ev.spans, pool: ev.pool, poolTokens: ev.result.pool_tokens }))
              break
            case 'error':
              streamError = true
              setRun((r) => ({ ...r, status: 'error', error: ev.message }))
              break
          }
        },
        ctrl.signal,
      )
      if (final) {
        const res: ResultEvent = final
        setActiveId(addHistory(request, res))
        const pm = pubmedFromSpans(res.spans)
        if (pm) setPubmed(pm)
      } else if (!streamError) {
        setRun((r) => ({ ...r, status: 'error', error: 'Stream ended without a result.' }))
      }
    } catch (e: unknown) {
      if (abortRef.current !== ctrl) return
      setRun((r) => ({ ...r, status: 'error', error: ctrl.signal.aborted ? 'Cancelled.' : errText(e) }))
    }
  }, [form, settings, addHistory])

  const runCompare = useCallback(async () => {
    const base = buildRequest(form, settings)
    compareAbortRef.current?.abort()
    const ctrl = new AbortController()
    compareAbortRef.current = ctrl
    const gates = config?.gates ?? ALL_GATES
    let rows: CompareRow[] = gates.map((g) => ({ gate: g, status: 'pending' }))
    const update = (g: GateName, patch: Partial<CompareRow>) => {
      rows = rows.map((r) => (r.gate === g ? { ...r, ...patch } : r))
      setCompare(rows)
    }
    setCompare(rows)
    setComparing(true)
    for (const gate of gates) {
      if (ctrl.signal.aborted) break
      update(gate, { status: 'running' })
      try {
        const res = await ask({ ...base, gate }, ctrl.signal)
        update(gate, { status: 'done', res })
        const pm = pubmedFromSpans(res.spans)
        if (pm) setPubmed(pm)
      } catch (e: unknown) {
        update(gate, { status: 'error', error: ctrl.signal.aborted ? 'Cancelled' : errText(e) })
      }
    }
    rows.filter((r) => r.status === 'pending').forEach((r) => update(r.gate, { status: 'error', error: 'Cancelled' }))
    setComparing(false)
  }, [form, settings, config])

  const showResult = (request: AskRequest, res: ResultEvent) => {
    abortRef.current?.abort()
    abortRef.current = null
    setRun({ ...IDLE, status: 'done', request, spans: res.spans, pool: res.pool, poolTokens: res.result.pool_tokens, result: res })
  }

  const restore = (e: HistoryEntry) => {
    const q = e.request
    setActiveId(e.id)
    setForm(formFromRequest(q))
    setSettings((s) => ({
      gate: q.gate,
      answer_model: q.answer_model ?? s.answer_model,
      probe_model: q.probe_model ?? s.probe_model,
      scout_model: q.scout_model ?? '',
      verifier_model: q.verifier_model ?? '',
      k_textbook: q.k_textbook,
      k_pubmed: q.k_pubmed,
      rewrite_query: q.rewrite_query,
      speculative_filter: q.speculative_filter,
      overlap_retrieval: q.overlap_retrieval,
      tau_m: q.tau_m ?? s.tau_m,
      tau_g: q.tau_g ?? s.tau_g,
      tau_sem: q.tau_sem,
    }))
    showResult(q, e.result)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const running = run.status === 'running'
  const header = <Header health={health} healthError={healthError} pubmed={pubmed.status} pubmedNote={pubmed.note} route={route} />
  if (route === '#/audit' || route.startsWith('#/replay/audit'))
    return (
      <div className="app">
        {header}
        <AuditPage replay={route !== '#/audit'} openId={route.split('/')[3] || null} />
      </div>
    )
  if (route.startsWith('#/replay'))
    return (
      <div className="app">
        {header}
        <ReplayPage caseKey={route.split('/')[2] || null} />
      </div>
    )
  const elapsed = running ? Math.max(0, now - run.t0) : 0
  const res = run.result

  return (
    <div className="app">
      {header}
      <main className="layout">
        <aside className="left">
          <Composer
            form={form}
            onChange={setForm}
            onSubmit={() => void submit()}
            onCompare={() => void runCompare()}
            onCancel={() => abortRef.current?.abort()}
            running={running}
            comparing={comparing}
          />
          <SettingsPanel config={config} settings={settings} onChange={setSettings} isMcq={form.mcq} disabled={running || comparing} />
          <HistoryList
            entries={history}
            activeId={activeId}
            onSelect={restore}
            onClear={() => setHistory(saveHistory([]))}
            onRemove={(id) => setHistory((h) => saveHistory(h.filter((x) => x.id !== id)))}
          />
        </aside>

        <div className="right">
          {run.status === 'error' && run.error && (
            <div className="alert error" role="alert">
              <strong>Request failed.</strong> {run.error}
            </div>
          )}
          {run.request && run.status !== 'idle' && (
            <div className="question-echo">
              <span className="eyebrow">Question</span>
              <p>{run.request.question}</p>
              {run.request.options && (
                <ol className="echo-options">
                  {Object.entries(run.request.options).map(([k, v]) => (
                    <li key={k} className={res?.result.answer === k ? 'chosen' : ''}>
                      <b>{k}.</b> {v}
                    </li>
                  ))}
                </ol>
              )}
            </div>
          )}

          {(run.status !== 'idle' || !compare) && (
            <DecisionCard res={res} request={run.request} start={run.start} running={running} stage={stageText(run)} />
          )}
          {res && <CostStrip result={res.result} summary={res.summary} spans={res.spans} />}
          {compare && (
            <CompareGates
              rows={compare}
              running={comparing}
              onCancel={() => compareAbortRef.current?.abort()}
              onClose={() => setCompare(null)}
              onView={(r) => {
                setActiveId(null)
                showResult({ ...buildRequest(form, settings), gate: r.result.gate }, r)
              }}
            />
          )}
          {run.status !== 'idle' && <Timeline spans={run.spans} running={running} elapsedMs={elapsed} wallMs={res?.summary.wall_ms ?? null} />}
          {run.status !== 'idle' && (
            <EvidencePanel
              pool={run.pool}
              poolTokens={run.poolTokens}
              result={res?.result ?? null}
              running={running}
              closedBookGate={run.request?.gate === 'closed_book'}
            />
          )}
        </div>
      </main>
      <footer className="footer muted small">
        MUG research prototype · evidence from MedRAG textbooks and PubMed (NCBI E-utilities) · answers may be wrong — always consult primary
        sources and a qualified clinician.
      </footer>
    </div>
  )
}
