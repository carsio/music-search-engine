import { Link } from "react-router-dom";
import type { SearchResultItem } from "../types";

interface Props {
  items: SearchResultItem[];
  showScores?: boolean;
}

function targetFor(item: SearchResultItem): string | null {
  if (item.intent === "track") return `/song/${item.id}`;
  if (item.intent === "artist") return `/artist/${item.id}`;
  return null;
}

export function ResultsList({ items, showScores = true }: Props) {
  if (items.length === 0) {
    return <div className="empty">Sem resultados para essa busca.</div>;
  }
  return (
    <ul className="results">
      {items.map((item) => {
        const href = targetFor(item);
        const inner = (
          <>
            <div className="result-rank">#{item.rank}</div>
            <div className="result-body">
              <div className="result-title">{item.title}</div>
              {item.subtitle && <div className="result-sub">{item.subtitle}</div>}
              {item.snippet && <p className="result-snippet">{item.snippet}</p>}
            </div>
            {showScores && (
              <div className="result-score">
                <span className="result-score-label">score</span>
                <span className="result-score-value">{item.score.toFixed(2)}</span>
              </div>
            )}
          </>
        );
        return (
          <li key={item.id} className={`result result--${item.intent}`}>
            {href ? <Link to={href}>{inner}</Link> : inner}
          </li>
        );
      })}
    </ul>
  );
}
