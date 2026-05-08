// Results page — chooses layout based on classification

const { ARTISTS: A2, SONGS: S2, WEB_RESULTS: W2, LYRIC_MATCHES: L2 } = window.MUSIC_DATA;
const { ArtistPanel, SongPanel, WebResult, LyricMatch } = window;

function ResultsArtist({ artist, accent, showScores, onSearch }) {
  const links = W2[artist.id] || [];
  return (
    <div className="results-layout results-2col" data-screen-label={`Result · ${artist.name}`}>
      <div className="results-main">
        <div className="results-eyebrow">Resultados para <strong>{artist.name}</strong></div>

        <div className="ri-explain">
          <div className="ri-explain-title">Por que esse resultado?</div>
          <div className="ri-explain-tags">
            <span className="ri-tag" style={{ borderColor: accent }}>match exato em <code>artistas.nome</code></span>
            <span className="ri-tag">boost: documento canônico</span>
            <span className="ri-tag">5 fontes externas</span>
          </div>
        </div>

        <div className="results-section-head">
          <h3 className="results-section-title">Da web</h3>
          {showScores && <span className="results-section-meta">ordenado por BM25</span>}
        </div>

        <div className="web-results">
          {links.map((r, i) => <WebResult key={i} result={r} accent={accent} showScores={showScores} />)}
        </div>

        <div className="results-section-head">
          <h3 className="results-section-title">Você também pode procurar por</h3>
        </div>
        <div className="related-row">
          {artist.topTracks.slice(0, 4).map(t => (
            <button key={t.title} className="related-chip" onClick={() => onSearch(t.title)}>
              <span className="related-chip-eyebrow">música</span>
              <span className="related-chip-title">{t.title}</span>
              <span className="related-chip-sub">{t.plays} plays</span>
            </button>
          ))}
        </div>
      </div>

      <aside className="results-side">
        <ArtistPanel artist={artist} accent={accent} onTrackClick={onSearch} />
      </aside>
    </div>
  );
}

function ResultsSong({ song, query, accent, showScores, onSearch }) {
  const artist = A2[song.artistId];
  return (
    <div className="results-layout results-2col" data-screen-label={`Result · ${song.title}`}>
      <div className="results-main">
        <div className="results-eyebrow">Letra de <strong>{song.title}</strong> · {song.artist}</div>
        <SongPanel song={song} query={query} accent={accent} />

        <div className="results-section-head" style={{ marginTop: 28 }}>
          <h3 className="results-section-title">Outras versões / fontes</h3>
        </div>
        <div className="web-results">
          <WebResult accent={accent} showScores={showScores} result={{
            site: "letras.mus.br",
            url: `letras.mus.br › ${artist.id} › ${song.id}`,
            title: `${song.title} - ${song.artist} - LETRAS.MUS.BR`,
            snippet: `Letra completa de "${song.title}" do álbum ${song.album}. Cifras, tradução e clipe oficial. ${song.plays} visualizações.`,
            score: 0.96,
          }} />
          <WebResult accent={accent} showScores={showScores} result={{
            site: "vagalume.com.br",
            url: `vagalume.com.br › ${artist.id} › ${song.id}`,
            title: `${song.title} – ${song.artist} (com cifra)`,
            snippet: `Cifra simplificada e letra de "${song.title}" — ${song.artist}. Tom: G · Capotraste: 2ª casa · ${song.year}.`,
            score: 0.83,
          }} />
          <WebResult accent={accent} showScores={showScores} result={{
            site: "youtube.com",
            url: `youtube.com › watch › ${song.id}`,
            title: `${song.artist} - ${song.title} (Clipe Oficial)`,
            snippet: `Vídeo oficial de "${song.title}". Inscreva-se no canal e ative o sininho. ${song.plays} de visualizações.`,
            score: 0.78,
          }} />
        </div>
      </div>

      <aside className="results-side">
        <div className="kp-card">
          <div className="kp-mini-header" style={{ background: artist.color }}>
            <span className="kp-placeholder-tag">FOTO · {artist.name.toUpperCase()}</span>
          </div>
          <div className="kp-body">
            <div className="kp-eyebrow">Artista</div>
            <h3 className="kp-title-sm">{artist.name}</h3>
            <p className="kp-bio kp-bio-sm">{artist.bio}</p>
            <div className="kp-mini-stats">
              <div><div className="kp-meta-label">Ouvintes/mês</div><div className="kp-meta-value">{artist.monthlyListeners}</div></div>
              <div><div className="kp-meta-label">Início</div><div className="kp-meta-value">{artist.yearStarted}</div></div>
            </div>
            <button className="kp-cta" style={{ background: accent }} onClick={() => onSearch(artist.name)}>
              Ver tudo de {artist.name} →
            </button>
            <div className="kp-section">
              <div className="kp-section-title">Mais músicas</div>
              <ol className="kp-track-list kp-track-list-mini">
                {artist.topTracks.filter(t => t.title !== song.title).slice(0, 4).map((t, i) => (
                  <li key={t.title} className="kp-track" onClick={() => onSearch(t.title)}>
                    <span className="kp-track-num">{String(i + 1).padStart(2, "0")}</span>
                    <span className="kp-track-title">{t.title}</span>
                    <span className="kp-track-plays">{t.plays}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}

function ResultsLyric({ token, matches, query, accent, showScores, onSearch }) {
  return (
    <div className="results-layout results-1col" data-screen-label={`Result · letras com "${token}"`}>
      <div className="results-main">
        <div className="results-eyebrow">Letras com o termo <strong>"{token}"</strong></div>

        <div className="ri-explain">
          <div className="ri-explain-title">Busca em corpus de letras</div>
          <div className="ri-explain-tags">
            <span className="ri-tag" style={{ borderColor: accent }}>full-text · letras_mus_br</span>
            <span className="ri-tag">stemmer · pt-BR</span>
            <span className="ri-tag">stop-words removidas</span>
            <span className="ri-tag">{matches.reduce((s, m) => s + m.snippets.length, 0)} ocorrências</span>
          </div>
        </div>

        <div className="results-section-head">
          <h3 className="results-section-title">{matches.length} músicas encontradas</h3>
          {showScores && <span className="results-section-meta">ranqueadas por TF-IDF</span>}
        </div>

        <div className="lyric-matches">
          {matches.map((m, i) => (
            <LyricMatch key={i} match={m} accent={accent} showScores={showScores}
              onClick={() => S2[m.songId] && onSearch(m.title)} />
          ))}
        </div>

        <div className="results-section-head" style={{ marginTop: 32 }}>
          <h3 className="results-section-title">Refine a busca</h3>
        </div>
        <div className="related-row">
          <button className="related-chip" onClick={() => onSearch(`"${token}" forró`)}>
            <span className="related-chip-eyebrow">filtrar</span>
            <span className="related-chip-title">"{token}" + forró</span>
          </button>
          <button className="related-chip" onClick={() => onSearch(`"${token}" mpb`)}>
            <span className="related-chip-eyebrow">filtrar</span>
            <span className="related-chip-title">"{token}" + MPB</span>
          </button>
          <button className="related-chip" onClick={() => onSearch("saudade")}>
            <span className="related-chip-eyebrow">termo similar</span>
            <span className="related-chip-title">saudade</span>
          </button>
        </div>
      </div>
    </div>
  );
}

function ResultsGenre({ genre, accent, showScores, onSearch }) {
  const artistsInGenre = Object.values(A2).filter(a => a.genres.some(g => g.toLowerCase().includes(genre)));
  return (
    <div className="results-layout results-1col" data-screen-label={`Result · gênero ${genre}`}>
      <div className="results-main">
        <div className="results-eyebrow">Gênero · <strong style={{ textTransform: "capitalize" }}>{genre}</strong></div>

        <div className="genre-hero" style={{ background: `linear-gradient(110deg, ${accent}, ${accent}55)` }}>
          <div className="genre-hero-inner">
            <h2 className="genre-title" style={{ textTransform: "capitalize" }}>{genre}</h2>
            <p className="genre-desc">Gênero musical brasileiro com forte presença no Nordeste. {artistsInGenre.length} artistas no índice, mais de 1.200 letras catalogadas.</p>
            <div className="genre-stats">
              <span><strong>{artistsInGenre.length}</strong> artistas</span>
              <span><strong>1.247</strong> músicas</span>
              <span><strong>89</strong> álbuns</span>
            </div>
          </div>
        </div>

        <div className="results-section-head">
          <h3 className="results-section-title">Artistas do gênero</h3>
        </div>
        <div className="artist-grid">
          {artistsInGenre.map(a => (
            <button key={a.id} className="artist-card" onClick={() => onSearch(a.name)}>
              <div className="artist-card-cover" style={{ background: a.color }}>
                <span className="kp-placeholder-tag">FOTO</span>
              </div>
              <div className="artist-card-info">
                <div className="artist-card-name">{a.name}</div>
                <div className="artist-card-meta">{a.origin} · desde {a.yearStarted}</div>
                <div className="artist-card-meta artist-card-meta-mono">{a.monthlyListeners} ouvintes/mês</div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function ResultsNoResults({ q, accent, onSearch }) {
  // Find a "did you mean" suggestion
  const allTerms = [
    ...Object.values(A2).map(a => a.name),
    ...Object.values(S2).map(s => s.title),
  ];
  const suggestion = allTerms.find(t => {
    const a = t.toLowerCase(), b = q.toLowerCase();
    return a.includes(b.slice(0, 4)) || b.includes(a.slice(0, 4));
  });

  return (
    <div className="results-layout results-1col" data-screen-label="Sem resultados">
      <div className="results-main">
        <div className="no-results">
          <div className="no-results-icon" style={{ background: accent }}>
            <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="#000" strokeWidth="2">
              <circle cx="11" cy="11" r="7" />
              <path d="m21 21-4.3-4.3" strokeLinecap="round" />
              <path d="M8 11h6" strokeLinecap="round" />
            </svg>
          </div>
          <h2 className="no-results-title">Nada encontrado para <em>"{q}"</em></h2>
          <p className="no-results-sub">Sua busca não retornou nenhum documento no índice <code>letras_mus_br</code>.</p>

          {suggestion && (
            <div className="did-you-mean">
              <span>Você quis dizer:</span>
              <button className="did-you-mean-link" style={{ color: accent }} onClick={() => onSearch(suggestion)}>
                {suggestion}?
              </button>
            </div>
          )}

          <div className="no-results-tips">
            <div className="no-results-tips-title">Dicas:</div>
            <ul>
              <li>Verifique a ortografia (acentos contam)</li>
              <li>Tente palavras-chave mais gerais (ex: <button className="inline-link" onClick={() => onSearch("forró")}>forró</button>)</li>
              <li>Busque por trechos de letra entre aspas</li>
              <li>Use <code>letra de [música]</code> para buscar letra específica</li>
            </ul>
          </div>

          <div className="no-results-debug">
            <div className="no-results-debug-row"><span>tokens analisados</span><code>[{q.split(/\s+/).map(t => `"${t}"`).join(", ")}]</code></div>
            <div className="no-results-debug-row"><span>após stemming</span><code>[{q.split(/\s+/).map(t => `"${t.toLowerCase().slice(0, -1)}"`).join(", ")}]</code></div>
            <div className="no-results-debug-row"><span>documentos varridos</span><code>14.382</code></div>
            <div className="no-results-debug-row"><span>tempo</span><code>3ms</code></div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ResultsAlbum({ accent, onSearch }) {
  const artist = A2["joao-gomes"];
  const album = artist.albums.find(a => a.title === "Raiz");
  return (
    <div className="results-layout results-2col" data-screen-label="Result · álbum">
      <div className="results-main">
        <div className="results-eyebrow">Álbum · <strong>{album.title}</strong> · {artist.name}</div>

        <div className="album-hero">
          <div className="album-hero-cover" style={{ background: `linear-gradient(135deg, ${artist.color}, ${accent})` }}>
            <div className="album-hero-cover-text">
              <div style={{ fontSize: 12, opacity: 0.7, letterSpacing: 1 }}>{artist.name.toUpperCase()}</div>
              <div style={{ fontSize: 28, fontFamily: "Instrument Serif, serif" }}>{album.title}</div>
              <div style={{ fontSize: 12, opacity: 0.7 }}>{album.year}</div>
            </div>
          </div>
          <div className="album-hero-info">
            <h2 className="album-hero-title">{album.title}</h2>
            <div className="album-hero-byline">
              <button className="album-hero-artist" onClick={() => onSearch(artist.name)}>{artist.name}</button>
              <span>· Álbum · {album.year} · {album.tracks} faixas</span>
            </div>
            <p className="album-hero-desc">Terceiro álbum de estúdio do cantor pernambucano, lançado em 2024. Volta às raízes do forró com produção contemporânea.</p>
            <div className="album-pills">
              <span className="kp-pill">Forró</span>
              <span className="kp-pill">Piseiro</span>
              <span className="kp-pill">2024</span>
              <span className="kp-pill">Estúdio</span>
            </div>
          </div>
        </div>

        <div className="results-section-head">
          <h3 className="results-section-title">Faixas</h3>
        </div>
        <ol className="album-track-list">
          {[
            { n: 1, t: "Pra Que Fui Me Apaixonar", d: "3:24", p: "89M", clickable: true },
            { n: 2, t: "Dengo", d: "2:58", p: "64M" },
            { n: 3, t: "Coração Cigano", d: "3:12", p: "41M" },
            { n: 4, t: "Saudade da Minha Terra", d: "4:02", p: "38M" },
            { n: 5, t: "Pé de Serra", d: "3:18", p: "29M" },
            { n: 6, t: "Mulher Encantada", d: "3:45", p: "22M" },
          ].map(f => (
            <li key={f.n} className="album-track" onClick={() => f.clickable && onSearch(f.t)}>
              <span className="album-track-num">{String(f.n).padStart(2, "0")}</span>
              <span className="album-track-title">{f.t}</span>
              <span className="album-track-plays">{f.p}</span>
              <span className="album-track-dur">{f.d}</span>
            </li>
          ))}
        </ol>
      </div>

      <aside className="results-side">
        <div className="kp-card">
          <div className="kp-body">
            <div className="kp-eyebrow">Sobre o artista</div>
            <h3 className="kp-title-sm">{artist.name}</h3>
            <p className="kp-bio kp-bio-sm">{artist.bio}</p>
            <button className="kp-cta" style={{ background: accent }} onClick={() => onSearch(artist.name)}>
              Página de {artist.name} →
            </button>
          </div>
        </div>
        <div className="kp-card kp-card-meta">
          <div className="kp-body">
            <div className="kp-eyebrow">Metadados RI</div>
            <div className="kp-meta-row"><span>doc_id</span><code>album_raiz_2024</code></div>
            <div className="kp-meta-row"><span>tipo</span><code>album</code></div>
            <div className="kp-meta-row"><span>n_faixas</span><code>14</code></div>
            <div className="kp-meta-row"><span>BM25</span><code>0.97</code></div>
            <div className="kp-meta-row"><span>indexed</span><code>2024-08-12</code></div>
          </div>
        </div>
      </aside>
    </div>
  );
}

Object.assign(window, { ResultsArtist, ResultsSong, ResultsLyric, ResultsGenre, ResultsNoResults, ResultsAlbum });
