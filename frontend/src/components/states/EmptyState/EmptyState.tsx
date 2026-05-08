import { Chip } from "../../primitives/Chip";
import styles from "./EmptyState.module.css";

interface EmptyStateProps {
  title: string;
  hint?: string;
  suggestions?: string[];
  onPick?: (q: string) => void;
}

export function EmptyState({ title, hint, suggestions, onPick }: EmptyStateProps) {
  return (
    <div className={styles.box}>
      <div className={styles.iconBox} aria-hidden>?</div>
      <h2 className={styles.title}>{title}</h2>
      {hint ? <p className={styles.hint}>{hint}</p> : null}
      {suggestions && suggestions.length > 0 ? (
        <>
          <div className={styles.subtitle}>Tente buscar por:</div>
          <div className={styles.chips}>
            {suggestions.map((s) => (
              <Chip key={s} onClick={() => onPick?.(s)}>
                {s}
              </Chip>
            ))}
          </div>
        </>
      ) : null}
    </div>
  );
}
