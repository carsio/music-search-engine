import { useMemo, useState } from "react";
import { highlight } from "../../../utils/highlight";
import styles from "./LyricsBlock.module.css";

interface LyricsBlockProps {
  lyrics?: string | null;
  query: string;
  collapsedLines?: number;
}

export function LyricsBlock({ lyrics, query, collapsedLines = 16 }: LyricsBlockProps) {
  const [expanded, setExpanded] = useState(false);

  const lines = useMemo(() => {
    if (!lyrics) return [];
    return lyrics.split("\n");
  }, [lyrics]);

  if (!lyrics) {
    return (
      <div className={styles.empty}>
        <p>Letra não disponível para esta música.</p>
      </div>
    );
  }

  const isTruncated = !expanded && lines.length > collapsedLines;
  const visibleLines = isTruncated ? lines.slice(0, collapsedLines) : lines;
  const hiddenCount = lines.length - collapsedLines;

  return (
    <div className={styles.wrap}>
      <div className={isTruncated ? styles.blockCollapsed : styles.block}>
        {visibleLines.map((line, i) => {
          const trimmed = line.trim();
          if (!trimmed) {
            return <div key={i} className={styles.blank} aria-hidden />;
          }
          return (
            <div key={i} className={styles.line}>
              <span className={styles.num}>{String(i + 1).padStart(2, "0")}</span>
              <span className={styles.text}>{highlight(line, query)}</span>
            </div>
          );
        })}
        {isTruncated ? <div className={styles.fade} aria-hidden /> : null}
      </div>
      {lines.length > collapsedLines ? (
        <button
          type="button"
          className={styles.toggle}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded
            ? "Mostrar menos"
            : `Mostrar mais (${hiddenCount} linhas)`}
        </button>
      ) : null}
    </div>
  );
}
