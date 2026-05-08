import { ResultsLayout } from "../components/layout/ResultsLayout";
import { ArtistPanel } from "../components/panels/ArtistPanel";
import { ResultsList } from "../components/panels/ResultsList";
import { Loading } from "../components/states/Loading";
import { Card } from "../components/primitives/Card";
import { Tag } from "../components/primitives/Tag";
import { Section } from "../components/layout/Section";
import { Chip } from "../components/primitives/Chip";
import type { ArtistResponse, SearchResponse } from "../api/types";

interface ArtistViewProps {
  query: string;
  response: SearchResponse;
  artist?: ArtistResponse;
  isEnriching: boolean;
  onSubmit: (q: string) => void;
}

export function ArtistView({ query, response, artist, isEnriching, onSubmit }: ArtistViewProps) {
  const items = response.items;
  const related = artist?.top_tracks.slice(0, 4) ?? [];

  const main = (
    <>
      <Card variant="dashed" padding="sm" style={{ marginBottom: 24 }}>
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 10,
            color: "var(--ink-3)",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            marginBottom: 8,
          }}
        >
          Por que esse resultado?
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          <Tag variant="ri">
            match em <code>artist.name</code>
          </Tag>
          <Tag variant="ri">
            algoritmo: <code>{response.algorithm}</code>
          </Tag>
          <Tag variant="ri">
            latência: <code>{response.elapsed_ms}ms</code>
          </Tag>
          {response.rerank_used ? <Tag variant="ri">LLM rerank</Tag> : null}
        </div>
      </Card>

      <Section
        title="Da web"
        meta={<>{items.length} resultados · BM25</>}
      >
        <ResultsList items={items} onPick={onSubmit} />
      </Section>

      {related.length > 0 ? (
        <Section title="Você também pode procurar por">
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
            {related.map((t) => (
              <Chip
                key={t.title}
                variant="related"
                eyebrow="música"
                sub={t.album ?? undefined}
                onClick={() => onSubmit(t.title)}
              >
                {t.title}
              </Chip>
            ))}
          </div>
        </Section>
      ) : null}
    </>
  );

  const sidebar = artist ? (
    <ArtistPanel artist={artist} onPick={onSubmit} />
  ) : isEnriching ? (
    <Loading variant="panel" />
  ) : (
    <ArtistPanel artist={fromTopItem(response)} onPick={onSubmit} />
  );

  return (
    <ResultsLayout
      eyebrow={
        <>
          Resultados para <strong>{query}</strong>
        </>
      }
      primary={main}
      secondary={sidebar}
    />
  );
}

function fromTopItem(response: SearchResponse): ArtistResponse {
  const top = response.items[0];
  const payload = (top?.payload ?? {}) as Record<string, unknown>;
  return {
    id: top?.id ?? "",
    name: top?.title ?? "",
    tagline: (top?.subtitle as string) ?? null,
    bio: (top?.snippet as string) ?? null,
    genres: Array.isArray(payload.genres) ? (payload.genres as string[]) : [],
    origin: (payload.origin as string) ?? null,
    year_started: (payload.year_started as number) ?? null,
    monthly_listeners: (payload.monthly_listeners as string) ?? null,
    popularity: (payload.popularity as number) ?? null,
    albums: [],
    top_tracks: [],
    source: null,
    source_url: null,
  };
}
