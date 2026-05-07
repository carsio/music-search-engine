import { useSearchParams } from "react-router-dom";
import { useState } from "react";
import { useLyricSearch, useSearch } from "../api/client";
import { ResultsList } from "../components/ResultsList";
import { LyricMatches } from "../components/LyricMatches";
import type { SearchAlgorithm } from "../types";

export function SearchResultsRoute() {
  const [params] = useSearchParams();
  const q = params.get("q") ?? "";
  const [algorithm, setAlgorithm] = useState<SearchAlgorithm>("bm25");
  const [rerank, setRerank] = useState(false);

  const search = useSearch(q, { algorithm, rerank });
  const lyric = useLyricSearch(q, algorithm);

  if (search.isPending && lyric.isPending) {
    return <div className="loading">Buscando...</div>;
  }
  if (search.isError) {
    return <div className="error">Erro: {String(search.error)}</div>;
  }

  return (
    <div className="search-results">
      <div className="search-meta">
        <h2 className="search-query">"{q}"</h2>
        <div className="search-controls">
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value as SearchAlgorithm)}
          >
            <option value="bm25">BM25</option>
            <option value="tfidf">TF-IDF</option>
          </select>
          <label className="toggle">
            <input
              type="checkbox"
              checked={rerank}
              onChange={(e) => setRerank(e.target.checked)}
            />{" "}
            LLM rerank
          </label>
        </div>
      </div>

      {search.data && (
        <div className="search-summary">
          intent: <strong>{search.data.intent_used}</strong> · algoritmo:{" "}
          <strong>{search.data.algorithm}</strong> · {search.data.elapsed_ms}ms
          {search.data.rerank_used && " · LLM-rerank"}
        </div>
      )}

      <section className="search-section">
        <h3>Resultados gerais</h3>
        {search.data && <ResultsList items={search.data.items} />}
      </section>

      <section className="search-section">
        <h3>Trechos de letra ({lyric.data?.matches.length ?? 0})</h3>
        {lyric.data && <LyricMatches matches={lyric.data.matches} query={q} />}
      </section>
    </div>
  );
}
