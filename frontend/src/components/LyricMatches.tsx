import { Link } from "react-router-dom";
import type { LyricMatch } from "../types";

interface Props {
  matches: LyricMatch[];
  query: string;
  showScores?: boolean;
}

function highlight(text: string, query: string) {
  if (!query) return text;
  const tokens = query
    .split(/\s+/)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .filter(Boolean);
  if (tokens.length === 0) return text;
  const pattern = new RegExp(`\\b(${tokens.join("|")})\\b`, "gi");
  const parts = text.split(pattern);
  return parts.map((part, i) =>
    pattern.test(part) ? (
      <mark key={i}>{part}</mark>
    ) : (
      <span key={i}>{part}</span>
    ),
  );
}

export function LyricMatches({ matches, query, showScores = true }: Props) {
  if (matches.length === 0) {
    return <div className="empty">Nenhum trecho de letra bateu com a busca.</div>;
  }
  return (
    <ol className="lyric-matches">
      {matches.map((match) => (
        <li key={match.song_id} className="lyric-match">
          <header>
            <Link to={`/song/${match.song_id}`} className="lyric-title">
              {match.title}
            </Link>
            <span className="lyric-artist">{match.artist}</span>
            {showScores && (
              <span className="lyric-score">
                BM25 <strong>{match.score.toFixed(2)}</strong>
              </span>
            )}
          </header>
          <ul className="lyric-snippets">
            {match.snippets.map((snip) => (
              <li key={snip.line}>
                <span className="lyric-snip-line">L{snip.line}</span>
                <span className="lyric-snip-text">{highlight(snip.text, query)}</span>
              </li>
            ))}
          </ul>
        </li>
      ))}
    </ol>
  );
}
