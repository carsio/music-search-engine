import { Link } from "react-router-dom";
import type { ArtistResponse } from "../types";

export function ArtistPanel({ artist }: { artist: ArtistResponse }) {
  return (
    <article className="artist-panel">
      <header>
        <h1>{artist.name}</h1>
        {artist.tagline && <p className="tagline">{artist.tagline}</p>}
        <div className="meta">
          {artist.origin && <span>📍 {artist.origin}</span>}
          {artist.year_started && <span>🎵 desde {artist.year_started}</span>}
          {artist.monthly_listeners && <span>👂 {artist.monthly_listeners}/mês</span>}
        </div>
        {artist.genres.length > 0 && (
          <ul className="genres">
            {artist.genres.map((g) => (
              <li key={g}>{g}</li>
            ))}
          </ul>
        )}
      </header>

      {artist.bio && <p className="bio">{artist.bio}</p>}

      {artist.top_tracks.length > 0 && (
        <section>
          <h2>Top tracks</h2>
          <ul className="top-tracks">
            {artist.top_tracks.map((t, i) => (
              <li key={`${t.title}-${i}`}>
                <span className="rank">#{i + 1}</span>
                <span className="title">{t.title}</span>
                {t.album && <span className="album">{t.album}</span>}
                {t.plays && <span className="plays">{t.plays}</span>}
              </li>
            ))}
          </ul>
        </section>
      )}

      {artist.albums.length > 0 && (
        <section>
          <h2>Discografia</h2>
          <ul className="albums">
            {artist.albums.map((a) => (
              <li key={a.title}>
                <span className="title">{a.title}</span>
                {a.year && <span className="year">{a.year}</span>}
                {a.tracks != null && <span className="tracks">{a.tracks} faixas</span>}
              </li>
            ))}
          </ul>
        </section>
      )}

      {artist.source_url && (
        <footer>
          <a href={artist.source_url} target="_blank" rel="noreferrer">
            Fonte: {artist.source ?? "wikipedia"}
          </a>{" "}
          · <Link to="/">voltar</Link>
        </footer>
      )}
    </article>
  );
}
