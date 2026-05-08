import { useSearchFlow } from "../hooks/useSearchFlow";
import { ArtistView } from "./ArtistView";
import { SongView } from "./SongView";
import { LyricView } from "./LyricView";
import { GenreView } from "./GenreView";
import { AlbumView } from "./AlbumView";
import { NoResultsView } from "./NoResultsView";
import { Loading } from "../components/states/Loading";
import { ErrorState } from "../components/states/ErrorState";
import { ResultsLayout } from "../components/layout/ResultsLayout";

interface ResultsRouterProps {
  query: string;
  onSubmit: (q: string) => void;
}

export function ResultsRouter({ query, onSubmit }: ResultsRouterProps) {
  const flow = useSearchFlow(query);

  if (flow.search.isLoading && !flow.search.data) {
    return (
      <ResultsLayout
        variant="one-col"
        primary={<Loading variant="panel" />}
      />
    );
  }

  if (flow.isError && !flow.search.data) {
    return (
      <ResultsLayout
        variant="one-col"
        primary={
          <ErrorState
            message={flow.error instanceof Error ? flow.error.message : "Falha na busca."}
            onRetry={() => flow.search.refetch()}
          />
        }
      />
    );
  }

  const response = flow.search.data;
  if (!response || response.items.length === 0) {
    return <NoResultsView query={query} onSubmit={onSubmit} />;
  }

  switch (flow.intent) {
    case "artist":
      return (
        <ArtistView
          query={query}
          response={response}
          artist={flow.artist}
          isEnriching={flow.isEnriching}
          onSubmit={onSubmit}
        />
      );
    case "track":
      return (
        <SongView
          query={query}
          response={response}
          song={flow.song}
          isEnriching={flow.isEnriching}
          onSubmit={onSubmit}
        />
      );
    case "lyric":
      return (
        <LyricView
          query={query}
          response={response}
          lyric={flow.lyric.data}
          isLoading={flow.lyric.isLoading}
          onSubmit={onSubmit}
        />
      );
    case "genre":
      return <GenreView query={query} response={response} onSubmit={onSubmit} />;
    case "album":
      return (
        <AlbumView
          query={query}
          response={response}
          album={flow.album}
          isEnriching={flow.isEnriching}
          onSubmit={onSubmit}
        />
      );
    default:
      return <NoResultsView query={query} onSubmit={onSubmit} />;
  }
}
