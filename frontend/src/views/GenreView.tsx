import { ResultsLayout } from "../components/layout/ResultsLayout";
import { GenrePanel } from "../components/panels/GenrePanel";
import type { SearchResponse } from "../api/types";

interface GenreViewProps {
  query: string;
  response: SearchResponse;
  onSubmit: (q: string) => void;
}

export function GenreView({ query, response, onSubmit }: GenreViewProps) {
  const top = response.items[0];
  const others = response.items.slice(1);

  return (
    <ResultsLayout
      variant="one-col"
      eyebrow={
        <>
          intent <strong>genre</strong> · query <strong>{query}</strong>
        </>
      }
      primary={
        <GenrePanel
          top={top}
          related={others}
          onPick={onSubmit}
        />
      }
    />
  );
}
