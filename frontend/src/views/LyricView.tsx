import { ResultsLayout } from "../components/layout/ResultsLayout";
import { LyricMatchCard } from "../components/panels/LyricMatchCard";
import { Loading } from "../components/states/Loading";
import { EmptyState } from "../components/states/EmptyState";
import { Card } from "../components/primitives/Card";
import { Tag } from "../components/primitives/Tag";
import type { LyricSearchResponse, SearchResponse } from "../api/types";

interface LyricViewProps {
  query: string;
  response: SearchResponse;
  lyric?: LyricSearchResponse;
  isLoading: boolean;
  onSubmit: (q: string) => void;
}

export function LyricView({ query, response, lyric, isLoading, onSubmit }: LyricViewProps) {
  const matches = lyric?.matches ?? [];

  const explainCard = (
    <Card variant="dashed" padding="sm">
      <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink-3)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
        Recuperação de informação
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        <Tag variant="ri">
          algoritmo: <code>{response.algorithm}</code>
        </Tag>
        <Tag variant="ri">índice: <code>letras_full_text</code></Tag>
        <Tag variant="ri">stemmer: <code>RSLP</code></Tag>
        <Tag variant="ri">stop-words: <code>pt-br</code></Tag>
        {lyric ? <Tag variant="ri">latência: <code>{lyric.elapsed_ms}ms</code></Tag> : null}
      </div>
    </Card>
  );

  let body: React.ReactNode;
  if (isLoading) {
    body = <Loading variant="list" />;
  } else if (matches.length === 0) {
    body = (
      <EmptyState
        title="Nenhum trecho de letra encontrado."
        hint="Tente uma palavra ou verso mais característico."
        suggestions={["coração", "amor", "saudade"]}
        onPick={onSubmit}
      />
    );
  } else {
    body = (
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {matches.map((m) => (
          <LyricMatchCard
            key={m.song_id}
            match={m}
            onPick={() => onSubmit(m.title)}
          />
        ))}
      </div>
    );
  }

  return (
    <ResultsLayout
      variant="one-col"
      eyebrow={
        <>
          intent <strong>lyric</strong> · query <strong>{query}</strong>
        </>
      }
      primary={
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {explainCard}
          {body}
        </div>
      }
    />
  );
}
