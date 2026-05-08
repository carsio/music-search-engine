import styles from "./ProgressBar.module.css";

interface ProgressBarProps {
  value: number | null | undefined;
  max?: number;
  accent?: string;
  showValue?: boolean;
}

export function ProgressBar({ value, max = 100, accent, showValue = true }: ProgressBarProps) {
  const v = value == null ? null : Math.max(0, Math.min(max, value));
  const pct = v == null ? 0 : (v / max) * 100;
  return (
    <div className={styles.row}>
      <div className={styles.bar} role="progressbar" aria-valuenow={v ?? undefined} aria-valuemin={0} aria-valuemax={max}>
        {v != null ? (
          <div
            className={styles.fill}
            style={{ width: `${pct}%`, background: accent ?? "var(--accent)" }}
          />
        ) : null}
      </div>
      {showValue ? (
        <span className={styles.num}>{v == null ? "—" : v}</span>
      ) : null}
    </div>
  );
}
