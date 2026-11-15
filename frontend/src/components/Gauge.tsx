interface Props {
  label: string
  value: number | null
  threshold?: number | null
  thresholdLabel?: string
  hint?: string
  tone?: 'move' | 'ground'
}

export function Gauge({ label, value, threshold, thresholdLabel, hint, tone = 'move' }: Props) {
  const v = value == null ? null : Math.max(0, Math.min(1, value))
  const passed = v != null && threshold != null ? v >= threshold : null
  return (
    <div className={`gauge ${tone}`}>
      <div className="gauge-head">
        <span className="gauge-label">{label}</span>
        <span className={`gauge-value ${passed === true ? 'pass' : passed === false ? 'fail' : ''}`}>
          {v == null ? '—' : v.toFixed(2)}
        </span>
      </div>
      <div className="gauge-track" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={v ?? undefined} aria-label={label}>
        <div className="gauge-fill" style={{ width: `${(v ?? 0) * 100}%` }} />
        {threshold != null && (
          <div className="gauge-threshold" style={{ left: `${threshold * 100}%` }} title={`${thresholdLabel ?? 'τ'} = ${threshold}`}>
            <span>
              {thresholdLabel ?? 'τ'} {threshold}
            </span>
          </div>
        )}
      </div>
      <div className="gauge-scale">
        <span>0</span>
        <span>{hint}</span>
        <span>1</span>
      </div>
    </div>
  )
}
