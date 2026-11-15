import type { HistoryEntry } from '../history'
import { GATE_INFO, truncate } from '../format'

interface Props {
  entries: HistoryEntry[]
  activeId: string | null
  onSelect: (e: HistoryEntry) => void
  onClear: () => void
  onRemove: (id: string) => void
}

function ago(ts: number): string {
  const s = Math.round((Date.now() - ts) / 1000)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return new Date(ts).toLocaleDateString()
}

export function HistoryList({ entries, activeId, onSelect, onClear, onRemove }: Props) {
  return (
    <section className="card history">
      <div className="card-title">
        <h2>History</h2>
        <div className="spacer" />
        {entries.length > 0 && (
          <button type="button" className="link-btn small" onClick={onClear}>
            Clear
          </button>
        )}
      </div>
      {entries.length === 0 ? (
        <p className="muted small empty">Previous questions are saved in this browser.</p>
      ) : (
        <ul>
          {entries.map((e) => {
            const r = e.result.result
            return (
              <li key={e.id} className={e.id === activeId ? 'active' : ''}>
                <button type="button" className="history-item" onClick={() => onSelect(e)}>
                  <span className={`dot ${r.admitted ? 'yes' : 'no'}`} title={r.admitted ? 'Context admitted' : 'Context withheld'} />
                  <span className="history-body">
                    <span className="history-q">{truncate(e.request.question, 110)}</span>
                    <span className="muted small">
                      {r.admitted ? 'Admitted' : 'Withheld'} · {GATE_INFO[r.gate]?.label ?? r.gate} · {truncate(r.answer_text || '—', 40)} · {ago(e.ts)}
                    </span>
                  </span>
                </button>
                <button type="button" className="icon-btn" aria-label="Remove from history" onClick={() => onRemove(e.id)}>
                  ×
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
