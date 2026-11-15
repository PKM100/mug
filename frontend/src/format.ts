export const LETTERS = ['A', 'B', 'C', 'D', 'E'] as const

export function fmtMs(ms: number): string {
  if (!Number.isFinite(ms)) return '—'
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`
}

export function fmtInt(n: number | null | undefined): string {
  return n == null ? '—' : Math.round(n).toLocaleString()
}

export function fmtNum(n: number | null | undefined, digits = 2): string {
  return n == null ? '—' : n.toFixed(digits)
}

export function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s
}

export function domId(id: string): string {
  return 'ev-' + id.replace(/[^A-Za-z0-9_-]/g, '_')
}

export function scrollToPassage(id: string): void {
  const el = document.getElementById(domId(id))
  if (!el) return
  el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  el.classList.remove('flash')
  void el.offsetWidth
  el.classList.add('flash')
}

export const GATE_INFO: Record<string, { label: string; blurb: string }> = {
  flip: {
    label: 'Flip (MUG default)',
    blurb: 'Admit evidence only if it flips the closed-book answer; then keep only the passages that carry the change and re-answer.',
  },
  grounded: {
    label: 'Grounded',
    blurb: 'Movement × groundedness: admit only if evidence moves the answer AND an entailment probe says it supports it.',
  },
  logit_shift: {
    label: 'Logit shift',
    blurb: 'Continuous movement: total-variation distance between closed-book and with-context answer distributions vs. τ_m.',
  },
  naive: { label: 'Naive RAG', blurb: 'Baseline: always answer with the full retrieved pool.' },
  closed_book: { label: 'Closed book', blurb: 'Baseline: never retrieve; answer from parametric knowledge only.' },
}

export const MODE_LABEL: Record<string, string> = {
  closed_book: 'closed-book',
  flip_filtered: 'flip → filtered',
  flip_full: 'flip → full pool',
  shift: 'shift',
  grounded_shift: 'grounded shift',
  ungrounded_keep: 'ungrounded → kept parametric',
  unverified_keep: 'unverified → kept parametric',
  naive: 'naive (full pool)',
  no_evidence: 'no evidence',
}
