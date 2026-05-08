// Painéis especializados (knowledge panels) — UI muda conforme o tipo de busca

const { useState, useEffect, useRef, useMemo } = React;

// ============ ARTIST KNOWLEDGE PANEL ============
function ArtistPanel({ artist, accent, onTrackClick }) {
  return (
    <div className="kp-card kp-artist">
      <div className="kp-artist-hero" style={{ background: artist.color }}>
        <div className="kp-artist-hero-overlay">
          <span className="kp-placeholder-tag">FOTO · {artist.name.toUpperCase()}</span>
        </div>
      </div>
      <div className="kp-body">
        <div className="kp-eyebrow">Artista · {artist.tagline}</div>
        <h2 className="kp-title">{artist.name}</h2>
        <p className="kp-bio">{artist.bio}</p>

        <div className="kp-meta-grid">
          <div className="kp-meta-item">
            <div className="kp-meta-label">Origem</div>
            <div className="kp-meta-value">{artist.origin}</div>
          </div>
          <div className="kp-meta-item">
            <div className="kp-meta-label">Início</div>
            <div className="kp-meta-value">{artist.yearStarted}</div>
          </div>
          <div className="kp-meta-item">
            <div className="kp-meta-label">Ouvintes/mês</div>
            <div className="kp-meta-value">{artist.monthlyListeners}</div>
          </div>
          <div className="kp-meta-item">
            <div className="kp-meta-label">Popularidade</div>
            <div className="kp-meta-value">
              <div className="kp-bar"><div className="kp-bar-fill" style={{ width: `${artist.popularity}%`, background: accent }} /></div>
              <span className="kp-bar-num">{artist.popularity}</span>
            </div>
          </div>
        </div>

        <div className="kp-section">
          <div className="kp-section-title">Gêneros</div>
          <div className="kp-pills">
            {artist.genres.map(g => <span key={g} className="kp-pill">{g}</span>)}
          </div>
        </div>

        <div className="kp-section">
          <div className="kp-section-title">Top músicas</div>
          <ol className="kp-track-list">
            {artist.topTracks.map((t, i) => (
              <li key={t.title} className="kp-track" onClick={() => onTrackClick && onTrackClick(t.title)}>
                <span className="kp-track-num">{String(i + 1).padStart(2, "0")}</span>
                <span className="kp-track-title">{t.title}</span>
                <span className="kp-track-album">{t.album}</span>
                <span className="kp-track-plays">{t.plays}</span>
              </li>
            ))}
          </ol>
        </div>

        <div className="kp-section">
          <div className="kp-section-title">Discografia</div>
          <div className="kp-album-row">
            {artist.albums.map(a => (
              <div key={a.title} className="kp-album">
                <div className="kp-album-cover" style={{ background: `linear-gradient(135deg, ${artist.color}, ${accent})` }}>
                  <span className="kp-album-year">{a.year}</span>
                </div>
                <div className="kp-album-title">{a.title}</div>
                <div className="kp-album-tracks">{a.tracks} faixas</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============ SONG / LYRIC PANEL ============
function SongPanel({ song, query, accent }) {
  // highlight query inside lyrics
  const lyricsLines = song.lyrics.split("\n");
  const highlightLine = (line) => {
    if (!query || query.length < 3) return line;
    const re = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
    const parts = line.split(re);
    return parts.map((p, i) =>
      re.test(p) ? <mark key={i} style={{ background: accent, color: "#000" }}>{p}</mark> : <React.Fragment key={i}>{p}</React.Fragment>
    );
  };

  return (
    <div className="kp-card kp-song">
      <div className="song-header" style={{ background: `linear-gradient(135deg, ${accent}, ${accent}aa)` }}>
        <div className="song-header-inner">
          <div className="song-eyebrow">Letra · Música</div>
          <h2 className="song-title">{song.title}</h2>
          <div className="song-byline">por <strong>{song.artist}</strong></div>
        </div>
      </div>

      <div className="song-meta-bar">
        <div className="song-meta-cell">
          <div className="song-meta-label">Álbum</div>
          <div className="song-meta-value">{song.album}</div>
        </div>
        <div className="song-meta-cell">
          <div className="song-meta-label">Ano</div>
          <div className="song-meta-value">{song.year}</div>
        </div>
        <div className="song-meta-cell">
          <div className="song-meta-label">Duração</div>
          <div className="song-meta-value">{song.duration}</div>
        </div>
        <div className="song-meta-cell">
          <div className="song-meta-label">Plays</div>
          <div className="song-meta-value">{song.plays}</div>
        </div>
        <div className="song-meta-cell">
          <div className="song-meta-label">Compositor</div>
          <div className="song-meta-value">{song.composers.join(", ")}</div>
        </div>
      </div>

      <div className="song-lyrics">
        {lyricsLines.map((line, i) => (
          <div key={i} className={`lyric-line ${line.trim() === "" ? "lyric-blank" : ""}`}>
            <span className="lyric-num">{line.trim() === "" ? "" : String(i + 1).padStart(3, "0")}</span>
            <span className="lyric-text">{highlightLine(line)}</span>
          </div>
        ))}
      </div>

      <div className="song-footer">
        <span className="song-footer-tag">FONTE · letras_mus_br</span>
        <span className="song-footer-tag">DOC_ID · {song.id}</span>
        <span className="song-footer-tag">TOKENS · {song.lyrics.split(/\s+/).length}</span>
      </div>
    </div>
  );
}

// ============ WEB RESULT ITEM ============
function WebResult({ result, accent, showScores }) {
  return (
    <div className="web-result">
      <div className="web-result-meta">
        <span className="web-favicon" style={{ background: accent }}></span>
        <div className="web-result-url-stack">
          <div className="web-result-site">{result.site}</div>
          <div className="web-result-path">{result.url}</div>
        </div>
        {showScores && (
          <span className="web-result-score">BM25 · {result.score.toFixed(2)}</span>
        )}
      </div>
      <h3 className="web-result-title">{result.title}</h3>
      <p className="web-result-snippet">{result.snippet}</p>
    </div>
  );
}

// ============ LYRIC MATCH RESULT ============
function LyricMatch({ match, accent, showScores, onClick }) {
  return (
    <div className="lyric-match" onClick={onClick}>
      <div className="lyric-match-header">
        <div>
          <div className="lyric-match-title">{match.title}</div>
          <div className="lyric-match-artist">{match.artist}</div>
        </div>
        {showScores && (
          <div className="lyric-match-scores">
            <span className="score-tag" style={{ borderColor: accent }}>
              <span className="score-tag-label">TF-IDF</span>
              <span className="score-tag-value">{match.score.toFixed(2)}</span>
            </span>
            <span className="score-tag-mini">{match.snippets.length} hits</span>
          </div>
        )}
      </div>
      <div className="lyric-match-snippets">
        {match.snippets.map((s, i) => (
          <div key={i} className="lyric-snippet">
            <span className="lyric-snippet-line">L{s.line}</span>
            <span className="lyric-snippet-text" dangerouslySetInnerHTML={{ __html: s.text.replace(/<mark>/g, `<mark style="background:${accent};color:#000">`) }} />
          </div>
        ))}
      </div>
    </div>
  );
}

Object.assign(window, { ArtistPanel, SongPanel, WebResult, LyricMatch });
