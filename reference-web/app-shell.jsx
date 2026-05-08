// Main app — query router decides which UI mode to show
const { useState: useStateA, useEffect: useEffectA, useRef: useRefA, useMemo: useMemoA } = React;

const { ARTISTS, SONGS, WEB_RESULTS, LYRIC_MATCHES } = window.MUSIC_DATA;

// Query understanding — decide qual "card" mostrar
const stripDiacritics = (s) => s.normalize("NFD").replace(/\p{Diacritic}/gu, "");

function classifyQuery(q) {
  const norm = q.trim().toLowerCase();
  const normStripped = stripDiacritics(norm);
  if (!norm) return { kind: "empty" };

  // intent triggers
  if (norm.startsWith("letra ") || norm.startsWith("letra de ")) {
    const rest = norm.replace(/^letra (de )?/, "");
    return { kind: "song-intent", target: rest };
  }
  if (norm.startsWith("álbum ") || norm.startsWith("album ") || norm.startsWith("discografia ")) {
    return { kind: "album", target: norm.replace(/^(álbum|album|discografia) (de )?/, "") };
  }

  // exact artist matches
  for (const a of Object.values(ARTISTS)) {
    if (norm === a.name.toLowerCase()) return { kind: "artist", artist: a };
  }
  // exact song matches
  for (const s of Object.values(SONGS)) {
    if (norm === s.title.toLowerCase()) return { kind: "song", song: s };
    if (norm === `${s.title.toLowerCase()} ${s.artist.toLowerCase()}`) return { kind: "song", song: s };
  }
  // partial artist match
  for (const a of Object.values(ARTISTS)) {
    if (a.name.toLowerCase().includes(norm) || norm.includes(a.name.toLowerCase())) {
      return { kind: "artist", artist: a };
    }
  }
  // partial song match
  for (const s of Object.values(SONGS)) {
    if (s.title.toLowerCase().includes(norm) || norm.includes(s.title.toLowerCase())) {
      return { kind: "song", song: s };
    }
  }
  // lyric token match
  for (const token of Object.keys(LYRIC_MATCHES)) {
    if (normStripped.includes(token)) return { kind: "lyric", token, matches: LYRIC_MATCHES[token] };
  }
  // ambiguous / fallback
  if (norm === "forró" || norm === "forro" || norm === "mpb" || norm === "piseiro") {
    return { kind: "genre", genre: norm };
  }
  return { kind: "no-results", q: norm };
}

// Suggestions while typing
const SUGGESTIONS_POOL = [
  { type: "artista", text: "Calcinha Preta", to: "Calcinha Preta" },
  { type: "música", text: "Cobertor — Calcinha Preta", to: "Cobertor" },
  { type: "artista", text: "João Gomes", to: "João Gomes" },
  { type: "música", text: "Pra Que Fui Me Apaixonar — João Gomes", to: "Pra Que Fui Me Apaixonar" },
  { type: "letra", text: "letras com \"coração\"", to: "coração" },
  { type: "letra", text: "letras com \"saudade\"", to: "saudade" },
  { type: "gênero", text: "Forró eletrônico", to: "forró" },
  { type: "álbum", text: "álbum Raiz", to: "álbum Raiz" },
];

function getSuggestions(q) {
  const norm = q.trim().toLowerCase();
  if (!norm) return [];
  return SUGGESTIONS_POOL.filter(s =>
    s.text.toLowerCase().includes(norm) || s.to.toLowerCase().includes(norm)
  ).slice(0, 6);
}

// =================================================================
// HEADER (compact, when query is set)
// =================================================================
function SearchHeader({ query, setQuery, onSubmit, accent, stats, focused, setFocused }) {
  const [local, setLocal] = useStateA(query);
  useEffectA(() => setLocal(query), [query]);
  const suggestions = useMemoA(() => getSuggestions(local), [local]);

  return (
    <header className="header">
      <div className="header-row">
        <div className="logo" onClick={() => onSubmit("")}>
          <div className="logo-mark" style={{ background: accent }}>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#000" strokeWidth="2.5">
              <circle cx="6" cy="18" r="3" />
              <circle cx="18" cy="15" r="3" />
              <path d="M9 18V5l12-2v12" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <span className="logo-text">músicabr<span style={{ color: accent }}>.</span></span>
        </div>

        <form className="search-form-compact" onSubmit={e => { e.preventDefault(); onSubmit(local); setFocused(false); }}>
          <div className={`search-box ${focused ? "focused" : ""}`} style={{ "--accent": accent }}>
            <svg className="search-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="7" />
              <path d="m21 21-4.3-4.3" strokeLinecap="round" />
            </svg>
            <input
              value={local}
              onChange={e => setLocal(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setTimeout(() => setFocused(false), 150)}
              placeholder="Pesquisar artista, música ou trecho de letra"
            />
            {local && (
              <button type="button" className="search-clear" onClick={() => setLocal("")}>×</button>
            )}
            <button type="submit" className="search-submit" style={{ background: accent }}>
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#000" strokeWidth="2.5">
                <path d="M5 12h14M13 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>

            {focused && suggestions.length > 0 && (
              <div className="suggestions">
                {suggestions.map((s, i) => (
                  <div key={i} className="suggestion" onMouseDown={() => { setLocal(s.to); onSubmit(s.to); setFocused(false); }}>
                    <span className={`sug-type sug-type-${s.type}`}>{s.type}</span>
                    <span className="sug-text">{s.text}</span>
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" className="sug-arrow">
                      <path d="M7 17 17 7M9 7h8v8" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                ))}
              </div>
            )}
          </div>
        </form>

        <div className="header-tabs">
          <span className="tab tab-active">Tudo</span>
          <span className="tab">Artistas</span>
          <span className="tab">Músicas</span>
          <span className="tab">Letras</span>
          <span className="tab">Álbuns</span>
        </div>
      </div>

      {stats && (
        <div className="stats-bar">
          <span className="stat-item">
            <span className="stat-num">{stats.total.toLocaleString("pt-BR")}</span>
            <span className="stat-label">documentos</span>
          </span>
          <span className="stat-sep">·</span>
          <span className="stat-item">
            <span className="stat-num">{stats.time}</span>
            <span className="stat-label">ms</span>
          </span>
          <span className="stat-sep">·</span>
          <span className="stat-item">
            <span className="stat-label">índice</span>
            <span className="stat-num">letras_mus_br</span>
          </span>
          <span className="stat-sep">·</span>
          <span className="stat-item">
            <span className="stat-label">ranking</span>
            <span className="stat-num">BM25 + TF-IDF</span>
          </span>
        </div>
      )}
    </header>
  );
}

// =================================================================
// HOMEPAGE (empty state)
// =================================================================
function Homepage({ onSearch, accent }) {
  const [local, setLocal] = useStateA("");
  const [focused, setFocused] = useStateA(false);
  const suggestions = useMemoA(() => getSuggestions(local), [local]);

  const quick = ["Calcinha Preta", "João Gomes", "Cobertor", "saudade", "forró"];

  return (
    <div className="home" data-screen-label="01 Homepage">
      <div className="home-inner">
        <div className="home-logo">
          <div className="home-logo-mark" style={{ background: accent }}>
            <svg viewBox="0 0 24 24" width="44" height="44" fill="none" stroke="#000" strokeWidth="2.2">
              <circle cx="6" cy="18" r="3" />
              <circle cx="18" cy="15" r="3" />
              <path d="M9 18V5l12-2v12" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <h1 className="home-title">
            música<span style={{ fontStyle: "italic" }}>br</span><span style={{ color: accent }}>.</span>
          </h1>
          <p className="home-tagline">
            Buscador de música brasileira · <span className="home-tagline-mono">RI · BM25 · TF-IDF</span>
          </p>
        </div>

        <form className="home-search" onSubmit={e => { e.preventDefault(); onSearch(local); }}>
          <div className={`search-box search-box-lg ${focused ? "focused" : ""}`} style={{ "--accent": accent }}>
            <svg className="search-icon" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="7" />
              <path d="m21 21-4.3-4.3" strokeLinecap="round" />
            </svg>
            <input
              autoFocus
              value={local}
              onChange={e => setLocal(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setTimeout(() => setFocused(false), 150)}
              placeholder="O que você quer ouvir?"
            />
            <button type="submit" className="search-submit search-submit-lg" style={{ background: accent }}>
              Buscar
            </button>

            {focused && suggestions.length > 0 && (
              <div className="suggestions">
                {suggestions.map((s, i) => (
                  <div key={i} className="suggestion" onMouseDown={() => { setLocal(s.to); onSearch(s.to); }}>
                    <span className={`sug-type sug-type-${s.type}`}>{s.type}</span>
                    <span className="sug-text">{s.text}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </form>

        <div className="home-quick">
          <span className="home-quick-label">Tente:</span>
          {quick.map(q => (
            <button key={q} className="home-quick-chip" onClick={() => onSearch(q)}>{q}</button>
          ))}
        </div>

        <div className="home-stats">
          <div className="home-stat">
            <div className="home-stat-num" style={{ color: accent }}>14.382</div>
            <div className="home-stat-label">letras indexadas</div>
          </div>
          <div className="home-stat">
            <div className="home-stat-num" style={{ color: accent }}>2.104</div>
            <div className="home-stat-label">artistas</div>
          </div>
          <div className="home-stat">
            <div className="home-stat-num" style={{ color: accent }}>891</div>
            <div className="home-stat-label">álbuns</div>
          </div>
          <div className="home-stat">
            <div className="home-stat-num" style={{ color: accent }}>~12ms</div>
            <div className="home-stat-label">latência média</div>
          </div>
        </div>

        <div className="home-foot">
          <span>Projeto acadêmico · Recuperação de Informação · curated-br-lyrics-search</span>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { SearchHeader, Homepage, classifyQuery, getSuggestions });
