import { useEffect, useState, type ReactNode } from "react";
import { SearchBox } from "../SearchBox";
import styles from "./SearchHeader.module.css";

interface SearchHeaderProps {
  query: string;
  onSubmit: (q: string) => void;
  onLogoClick: () => void;
  stats?: { label: string; value: string }[];
  actions?: ReactNode;
}

export function SearchHeader({ query, onSubmit, onLogoClick, stats, actions }: SearchHeaderProps) {
  const [draft, setDraft] = useState(query);

  useEffect(() => {
    setDraft(query);
  }, [query]);

  return (
    <header className={styles.header}>
      <div className={styles.row}>
        <button className={styles.logo} onClick={onLogoClick} aria-label="Voltar à home">
          <span className={styles.logoMark} aria-hidden>
            <svg
              viewBox="0 0 24 24"
              width="20"
              height="20"
              fill="none"
              stroke="#000"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="6" cy="18" r="3" />
              <circle cx="18" cy="15" r="3" />
              <path d="M9 18V5l12-2v12" />
            </svg>
          </span>
          <span className={styles.logoText}>
            música<em className={styles.logoItalic}>br</em>
            <span className={styles.logoDot}>.</span>
          </span>
        </button>
        <div className={styles.searchWrap}>
          <SearchBox
            value={draft}
            onChange={setDraft}
            onSubmit={onSubmit}
            variant="header"
          />
        </div>
        {actions ? <div className={styles.actions}>{actions}</div> : null}
      </div>
      {stats && stats.length > 0 ? (
        <div className={styles.stats}>
          {stats.map((s, i) => (
            <span key={s.label} className={styles.statItem}>
              <span className={styles.statNum}>{s.value}</span>
              <span className={styles.statLabel}>{s.label}</span>
              {i < stats.length - 1 ? <span className={styles.sep}>·</span> : null}
            </span>
          ))}
        </div>
      ) : null}
    </header>
  );
}
