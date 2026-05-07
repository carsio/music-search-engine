import { Link } from "react-router-dom";
import type { SongResponse } from "../types";

export function SongPanel({ song, query }: { song: SongResponse; query?: string }) {
  return (
    <article className="song-panel">
      <header>
        <h1>{song.title}</h1>
        <p className="artist">{song.artist}</p>
        <div className="meta">
          {song.album && <span>💿 {song.album}</span>}
          {song.year && <span>{song.year}</span>}
          {song.duration && <span>{song.duration}</span>}
          {song.macro_genre && <span>{song.macro_genre}</span>}
        </div>
      </header>

      {song.lyrics && <Lyrics text={song.lyrics} query={query} />}

      {song.lyrics_source_url && (
        <footer>
          <a href={song.lyrics_source_url} target="_blank" rel="noreferrer">
            Letra: {song.lyrics_source ?? "fonte"}
          </a>{" "}
          · <Link to="/">voltar</Link>
        </footer>
      )}
    </article>
  );
}

function Lyrics({ text, query }: { text: string; query?: string }) {
  const lines = text.split("\n");
  const tokens =
    query
      ?.split(/\s+/)
      .map((t) => t.toLowerCase())
      .filter(Boolean) ?? [];
  return (
    <pre className="lyrics">
      {lines.map((line, i) => {
        const lower = line.toLowerCase();
        const hit = tokens.some((t) => lower.includes(t));
        return (
          <div key={i} className={`lyric-line ${hit ? "lyric-line--hit" : ""}`}>
            <span className="lineno">{i + 1}</span>
            <span className="text">{line || " "}</span>
          </div>
        );
      })}
    </pre>
  );
}
