import { Tag } from "../../primitives/Tag";
import { highlightHtml } from "../../../utils/highlight";
import type { LyricMatch } from "../../../api/types";
import styles from "./LyricMatchCard.module.css";

interface LyricMatchCardProps {
  match: LyricMatch;
  onPick: () => void;
  showScore?: boolean;
}

export function LyricMatchCard({ match, onPick, showScore = true }: LyricMatchCardProps) {
  return (
    <article className={styles.card} onClick={onPick} role="button" tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onPick();
        }
      }}
    >
      <header className={styles.header}>
        <div className={styles.titleBlock}>
          <h3 className={styles.title}>{match.title}</h3>
          <p className={styles.artist}>{match.artist}</p>
        </div>
        {showScore ? (
          <div className={styles.scores}>
            <Tag variant="score">BM25 · {match.score.toFixed(2)}</Tag>
            <span className={styles.snippetCount}>
              {match.snippets.length} trecho{match.snippets.length !== 1 ? "s" : ""}
            </span>
          </div>
        ) : null}
      </header>
      <div className={styles.snippets}>
        {match.snippets.map((s, i) => (
          <div key={i} className={styles.snippet}>
            <span className={styles.line}>L{s.line}</span>
            <span className={styles.text}>{highlightHtml(s.text)}</span>
          </div>
        ))}
      </div>
    </article>
  );
}
