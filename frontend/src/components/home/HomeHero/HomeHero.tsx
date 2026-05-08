import { useState } from "react";
import { SearchBox } from "../../search/SearchBox";
import styles from "./HomeHero.module.css";

interface HomeHeroProps {
  onSubmit: (q: string) => void;
}

function MusicNote({ size = 44 }: { size?: number }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="#000"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="6" cy="18" r="3" />
      <circle cx="18" cy="15" r="3" />
      <path d="M9 18V5l12-2v12" />
    </svg>
  );
}

export function HomeHero({ onSubmit }: HomeHeroProps) {
  const [draft, setDraft] = useState("");
  return (
    <section className={styles.hero}>
      <div className={styles.inner}>
        <div className={styles.logo}>
          <span className={styles.mark} aria-hidden>
            <MusicNote size={44} />
          </span>
          <h1 className={styles.title}>
            música<em className={styles.italic}>br</em>
            <span className={styles.dot}>.</span>
          </h1>
          <p className={styles.tagline}>
            Buscador de música brasileira
            <span className={styles.taglineMono}>RI · BM25 · TF-IDF</span>
          </p>
        </div>
        <div className={styles.search}>
          <SearchBox
            value={draft}
            onChange={setDraft}
            onSubmit={onSubmit}
            variant="hero"
            autoFocus
            ctaLabel="Buscar"
          />
        </div>
      </div>
    </section>
  );
}
