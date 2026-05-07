import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { SearchBox } from "./SearchBox";
import { fetchHealth } from "../api/client";

const SUGGESTIONS = [
  "amor saudade",
  "garota de ipanema",
  "anitta",
  "samba",
  "Caetano Veloso",
  "trecho coração",
];

export function Homepage() {
  const navigate = useNavigate();
  const [q, setQ] = useState("");
  const [health, setHealth] = useState<{
    tracks_indexed: number;
    entities: Record<string, number>;
    llm: boolean;
  } | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const submit = (value: string) => {
    if (!value.trim()) return;
    navigate(`/search?q=${encodeURIComponent(value)}`);
  };

  return (
    <section className="homepage">
      <h1 className="hero">
        <span className="hero-serif">músicabr</span>
      </h1>
      <p className="hero-sub">Buscador acadêmico de música brasileira — BM25, TF-IDF + LLM</p>

      <SearchBox value={q} onChange={setQ} onSubmit={submit} />

      <div className="suggestions">
        {SUGGESTIONS.map((s) => (
          <button key={s} className="chip" onClick={() => submit(s)}>
            {s}
          </button>
        ))}
      </div>

      {health && (
        <div className="health">
          <span>
            <strong>{health.tracks_indexed.toLocaleString("pt-BR")}</strong> letras indexadas
          </span>
          {Object.entries(health.entities).length > 0 && (
            <span>
              · entidades:{" "}
              {Object.entries(health.entities)
                .map(([k, v]) => `${k}:${v}`)
                .join(" ")}
            </span>
          )}
          <span>· LLM: {health.llm ? "ativa" : "fallback heurístico"}</span>
        </div>
      )}
    </section>
  );
}
