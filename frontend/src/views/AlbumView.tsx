import { ResultsLayout } from "../components/layout/ResultsLayout";
import { AlbumPanel, AlbumSidebar } from "../components/panels/AlbumPanel";
import { ResultsList } from "../components/panels/ResultsList";
import { Card } from "../components/primitives/Card";
import { Tag } from "../components/primitives/Tag";
import { Section } from "../components/layout/Section";
import { Loading } from "../components/states/Loading";
import type { AlbumResponse, SearchResponse } from "../api/types";

interface AlbumViewProps {
  query: string;
  response: SearchResponse;
  album?: AlbumResponse;
  isEnriching: boolean;
  onSubmit: (q: string) => void;
}

export function AlbumView({ query, response, album, isEnriching, onSubmit }: AlbumViewProps) {
  const top = response.items[0];
  const items = response.items;
  const detail = album ?? (!isEnriching ? fromTopItem(response) : undefined);

  const main = detail ? (
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
            match em <code>album.title</code>
          </Tag>
          <Tag variant="ri">
            algoritmo: <code>{response.algorithm}</code>
          </Tag>
          <Tag variant="ri">
            latência: <code>{response.elapsed_ms}ms</code>
          </Tag>
          {detail.album_type ? (
            <Tag variant="ri">tipo: <code>{detail.album_type}</code></Tag>
          ) : null}
        </div>
      </Card>

      {top ? <AlbumPanel album={detail} item={top} onPick={onSubmit} /> : null}

      {items.length > 1 ? (
        <Section
          title="Outros resultados"
          meta={<>{items.length - 1} resultados relacionados</>}
        >
          <ResultsList items={items.slice(1)} onPick={onSubmit} />
        </Section>
      ) : null}
    </>
  ) : (
    <Loading variant="panel" />
  );

  return (
    <ResultsLayout
      eyebrow={
        <>
          Resultados para <strong>{query}</strong>
        </>
      }
      primary={main}
      secondary={detail && top ? <AlbumSidebar album={detail} item={top} onPick={onSubmit} /> : undefined}
    />
  );
}

function fromTopItem(response: SearchResponse): AlbumResponse {
  const top = response.items[0];
  const payload = (top?.payload ?? {}) as Record<string, unknown>;
  const artist = (payload.artist as string) ?? top?.subtitle ?? "Artista desconhecido";
  const tags = Array.isArray(payload.tags) ? (payload.tags as string[]) : [];

  return {
    id: top?.id ?? "",
    title: top?.title ?? "",
    artist,
    artist_id: (payload.artist_id as string) ?? null,
    year: (payload.year as number) ?? null,
    description: (payload.description as string) ?? top?.snippet ?? null,
    tags,
    tracks_count: (payload.tracks_count as number) ?? null,
    cover_url: (payload.cover_url as string) ?? null,
    artist_image_url: (payload.artist_image_url as string) ?? null,
    album_type: (payload.album_type as string) ?? null,
    label: (payload.label as string) ?? null,
    duration: null,
    total_duration_ms: null,
    tracks: [],
    artist_summary: {
      id: ((payload.artist_id as string) ?? "") || "",
      name: artist,
      image_url: (payload.artist_image_url as string) ?? null,
      genres: tags,
      popularity: null,
      followers_total: null,
      top_tracks: [],
      albums: [],
    },
  };
}
