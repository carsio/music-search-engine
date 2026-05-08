import { ResultLink } from "../ResultLink";
import type { SearchResultItem } from "../../../api/types";
import styles from "./ResultsList.module.css";

interface ResultsListProps {
  items: SearchResultItem[];
  query?: string;
  onPick: (q: string) => void;
  showScore?: boolean;
}

export function ResultsList({ items, query, onPick, showScore }: ResultsListProps) {
  if (items.length === 0) return null;
  return (
    <div className={styles.list}>
      {items.map((item) => (
        <ResultLink
          key={item.id}
          item={item}
          query={query}
          onPick={onPick}
          showScore={showScore}
        />
      ))}
    </div>
  );
}
