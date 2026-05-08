import { ResultsLayout } from "../components/layout/ResultsLayout";
import { SongPanel } from "../components/panels/SongPanel";
import { ResultsList } from "../components/panels/ResultsList";
import { Loading } from "../components/states/Loading";
import { Card } from "../components/primitives/Card";
import { Button } from "../components/primitives/Button";
import { Section } from "../components/layout/Section";
import type { SearchResponse, SongResponse } from "../api/types";
import styles from "./SongView.module.css";

interface SongViewProps {
  query: string;
  response: SearchResponse;
  song?: SongResponse;
  isEnriching: boolean;
  onSubmit: (q: string) => void;
}

export function SongView({ query, response, song, isEnriching, onSubmit }: SongViewProps) {
  const otherResults = response.items.slice(1, 6);
  const main = song ? (
    <SongPanel song={song} query={query} />
  ) : isEnriching ? (
    <Loading variant="panel" />
  ) : (
    <SongPanel song={fromTopItem(response)} query={query} />
  );

  const artistName = song?.artist ?? (response.items[0]?.subtitle ?? "");

  const sidebar = (
    <div className={styles.side}>
      {artistName ? (
        <Card variant="flat" padding="md">
          <div className={styles.eyebrow}>Artista</div>
          <h3 className={styles.artistName}>{artistName}</h3>
          {song?.macro_genre || (song?.genres?.length ?? 0) > 0 ? (
            <p className={styles.artistMeta}>
              {[song?.macro_genre, ...(song?.genres ?? [])].filter(Boolean).slice(0, 3).join(" · ")}
            </p>
          ) : null}
          <Button
            variant="cta"
            onClick={() => onSubmit(artistName)}
            className={styles.cta}
          >
            Ver tudo de {artistName} →
          </Button>
        </Card>
      ) : null}

      {otherResults.length > 0 ? (
        <Section title="Outras músicas">
          <div className={styles.related}>
            {otherResults.map((item) => (
              <button
                key={item.id}
                className={styles.relatedItem}
                onClick={() => onSubmit(item.title)}
              >
                <span className={styles.relatedTitle}>{item.title}</span>
                {item.subtitle ? (
                  <span className={styles.relatedSub}>{item.subtitle}</span>
                ) : null}
              </button>
            ))}
          </div>
        </Section>
      ) : null}
    </div>
  );

  return (
    <ResultsLayout
      eyebrow={
        <>
          Letra de <strong>{song?.title ?? response.items[0]?.title}</strong>
          {artistName ? <> · {artistName}</> : null}
        </>
      }
      primary={
        <>
          {main}
          {response.items.length > 1 ? (
            <Section title="Outras versões e fontes" meta={<>{response.items.length - 1} resultados</>}>
              <ResultsList
                items={response.items.slice(1)}
                query={query}
                onPick={onSubmit}
              />
            </Section>
          ) : null}
        </>
      }
      secondary={sidebar}
    />
  );
}

function fromTopItem(response: SearchResponse): SongResponse {
  const top = response.items[0];
  const payload = (top?.payload ?? {}) as Record<string, unknown>;
  return {
    id: top?.id ?? "",
    title: top?.title ?? "",
    artist: (top?.subtitle as string) ?? "",
    album: (payload.album_name as string) ?? null,
    year: (payload.release_year as number) ?? null,
    duration: null,
    plays: null,
    composers: [],
    lyrics: (payload.lyrics_preview as string) ?? null,
    lyrics_source: (payload.lyrics_source as string) ?? null,
    lyrics_source_url: (payload.lyrics_source_url as string) ?? null,
    genres:
      typeof payload.artist_genres === "string"
        ? (payload.artist_genres as string).split(",").map((g) => g.trim()).filter(Boolean)
        : [],
    macro_genre: (payload.macro_genre as string) ?? null,
  };
}
