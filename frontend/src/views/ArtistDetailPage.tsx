import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useArtist } from "../api/hooks";
import { ResultsLayout } from "../components/layout/ResultsLayout";
import { ArtistPanel } from "../components/panels/ArtistPanel";
import { Loading } from "../components/states/Loading";
import { ErrorState } from "../components/states/ErrorState";
import { Button } from "../components/primitives/Button";
import { SearchHeader } from "../components/search/SearchHeader";
import styles from "./DetailPage.module.css";

export function ArtistDetailPage() {
  const { id = "" } = useParams();
  const [params] = useSearchParams();
  const query = params.get("q") ?? "";
  const navigate = useNavigate();
  const { data, isLoading, isError, error, refetch } = useArtist(id);

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
            <span>Artista · /artist/{id}</span>
          </span>
        }
        primary={
          isLoading ? (
            <Loading variant="panel" />
          ) : isError ? (
            <ErrorState
              message={error instanceof Error ? error.message : "Falha ao carregar artista."}
              onRetry={() => refetch()}
            />
          ) : data ? (
            <ArtistPanel artist={data} onPick={handleSearch} />
          ) : null
        }
      />
    </div>
  );
}
