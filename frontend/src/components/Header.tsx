import { STATIC, type Health } from '../api'

export type PubmedStatus = 'unknown' | 'ok' | 'error'

interface Props {
  health: Health | null
  healthError: string | null
  pubmed: PubmedStatus
  pubmedNote?: string
  route?: string
}

const isReplay = (route: string) => route.startsWith('#/replay')
const isAudit = (route: string) => route === '#/audit'

export function Header({ health, healthError, pubmed, pubmedNote, route = '' }: Props) {
  const tb = healthError
    ? { cls: 'bad', text: 'Backend unreachable', title: healthError }
    : !health
      ? { cls: 'idle', text: 'Checking…', title: '' }
      : health.textbooks_ready
        ? { cls: 'ok', text: `Textbooks · ${health.textbook_snippets.toLocaleString()} snippets`, title: 'MedRAG textbook index loaded' }
        : health.textbooks_error
          ? { cls: 'bad', text: 'Textbooks unavailable', title: health.textbooks_error }
          : { cls: 'warn', text: 'Textbook index loading…', title: 'Index is warming up' }
  const pm =
    healthError
      ? { cls: 'bad', text: 'PubMed', title: healthError }
      : pubmed === 'error'
        ? { cls: 'bad', text: 'PubMed · error', title: pubmedNote ?? '' }
        : pubmed === 'ok'
          ? { cls: 'ok', text: 'PubMed · live', title: pubmedNote ?? 'Last retrieval succeeded' }
          : { cls: 'idle', text: 'PubMed · live (E-utilities)', title: 'Queried live per request' }

  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="brand">
          <div className="logo" aria-hidden>
            M
          </div>
          <div>
            <h1>
              MUG <span className="dot">·</span> Marginal-Utility Gated Medical QA
            </h1>
            <p className="subtitle">
              Retrieves textbook &amp; PubMed evidence, then admits it only if it actually <em>moves</em> the answer.
            </p>
          </div>
        </div>
        {STATIC ? (
        <nav className="nav">
          <a href="#/replay" className={route.startsWith('#/replay/audit') ? '' : 'on'}>Replay</a>
          <a href="#/replay/audit" className={route.startsWith('#/replay/audit') ? 'on' : ''}>Audit log</a>
          <a href="../">About</a>
          <a href="https://github.com/PKM100/mug">GitHub</a>
        </nav>
        ) : (
        <nav className="nav">
          <a href="#/" className={isAudit(route) || isReplay(route) ? '' : 'on'}>Ask</a>
          <a href="#/audit" className={isAudit(route) ? 'on' : ''}>Audit</a>
          <a href="#/replay" className={isReplay(route) ? 'on' : ''} title="Recorded cases; no credentials or network needed">
            Replay
          </a>
        </nav>
        )}
        {isReplay(route) ? (
          <div className="health">
            <span className="pill replay" title="Recorded traces rendered offline; no backend or model calls">
              <i />
              Replay · recorded, offline
            </span>
          </div>
        ) : (
        <div className="health">
          <span className={`pill ${tb.cls}`} title={tb.title}>
            <i />
            {tb.text}
          </span>
          <span className={`pill ${pm.cls}`} title={pm.title}>
            <i />
            {pm.text}
          </span>
        </div>
        )}
      </div>
      <div className="disclaimer" role="note">
        <strong>Research prototype — not medical advice.</strong> Do not use for clinical decisions.
      </div>
    </header>
  )
}
