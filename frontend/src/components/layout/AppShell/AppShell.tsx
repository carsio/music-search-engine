import { useNavigate } from "react-router-dom";
import { useQueryParam } from "../../../hooks/useQueryParam";
import { useHealth } from "../../../api/hooks";
import { SearchHeader } from "../../search/SearchHeader";
import { HomeView } from "../../../views/HomeView";
import { ResultsRouter } from "../../../views/ResultsRouter";
import { formatNumber } from "../../../utils/format";
import styles from "./AppShell.module.css";

export function AppShell() {
  const [query, setQuery] = useQueryParam("q");
  const navigate = useNavigate();
  const health = useHealth();

  const stats = health.data
    ? [
        { label: "letras", value: formatNumber(health.data.tracks_indexed) },
        ...Object.entries(health.data.entities).map(([k, v]) => ({
          label: k,
          value: formatNumber(v),
        })),
        { label: "llm", value: health.data.llm ? "ativa" : "off" },
      ]
    : undefined;

  const hasQuery = query.trim().length > 0;

  return (
    <div className={styles.app}>
      {hasQuery ? (
        <SearchHeader
          query={query}
          onSubmit={setQuery}
          onLogoClick={() => navigate("/")}
          stats={stats}
        />
      ) : null}
      <main className={styles.main}>
        {hasQuery ? (
          <ResultsRouter query={query} onSubmit={setQuery} />
        ) : (
          <HomeView onSubmit={setQuery} />
        )}
      </main>
    </div>
  );
}
