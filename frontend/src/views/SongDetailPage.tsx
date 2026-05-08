import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useSong } from "../api/hooks";
import { ResultsLayout } from "../components/layout/ResultsLayout";
import { SongPanel } from "../components/panels/SongPanel";
import { Loading } from "../components/states/Loading";
import { ErrorState } from "../components/states/ErrorState";
import { Button } from "../components/primitives/Button";
import { SearchHeader } from "../components/search/SearchHeader";
import styles from "./DetailPage.module.css";

export function SongDetailPage() {
  const { id = "" } = useParams();
  const [params] = useSearchParams();
  const query = params.get("q") ?? "";
  const navigate = useNavigate();
  const { data, isLoading, isError, error, refetch } = useSong(id);

  function handleSearch(q: string) {
    navigate(`/?q=${encodeURIComponent(q)}`);
  }

  return (
    <div className={styles.page}>
      <SearchHeader
        query={query}
        onSubmit={handleSearch}
        onLogoClick={() => navigate("/")}
      />
      <ResultsLayout
        variant="one-col"
        eyebrow={
          <span className={styles.crumbRow}>
            <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
              ← voltar
            </Button>
            <span>Música · /song/{id}</span>
          </span>
        }
        primary={
          isLoading ? (
            <Loading variant="panel" />
          ) : isError ? (
            <ErrorState
              message={error instanceof Error ? error.message : "Falha ao carregar música."}
              onRetry={() => refetch()}
            />
          ) : data ? (
            <SongPanel song={data} query={query} />
          ) : null
        }
      />
    </div>
  );
}
