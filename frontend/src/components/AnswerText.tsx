import type { Candidate } from '../api'
import { scrollToPassage } from '../format'

interface Props {
  text: string
  /** Context the answer was generated with; [Doc n] → docs[n-1]. */
  docs: Candidate[]
}

export function AnswerText({ text, docs }: Props) {
  const parts = text.split(/(\[Docs?\s*\d+(?:\s*[,–-]\s*\d+)*\])/g)
  return (
    <div className="answer-text">
      {parts.map((part, i) => {
        const m = /^\[Docs?\s*([\d\s,–-]+)\]$/.exec(part)
        if (!m) return <span key={i}>{part}</span>
        const nums = m[1].split(/[,–-]/).map((s) => parseInt(s.trim(), 10)).filter((n) => !Number.isNaN(n))
        return (
          <span key={i} className="cites">
            {nums.map((n) => {
              const doc = docs[n - 1]
              return doc ? (
                <button key={n} type="button" className="cite" title={doc.title} onClick={() => scrollToPassage(doc.id)}>
                  Doc {n}
                </button>
              ) : (
                <span key={n} className="cite missing" title="Not in the admitted context">
                  Doc {n}
                </span>
              )
            })}
          </span>
        )
      })}
    </div>
  )
}
