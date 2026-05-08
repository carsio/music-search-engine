import { Tag } from "../../primitives/Tag";
import type { SearchResultItem } from "../../../api/types";
import { placeholder } from "../../../utils/format";
import styles from "./AlbumPanel.module.css";

interface AlbumPanelProps {
  item: SearchResultItem;
  onPick: (q: string) => void;
}

export function AlbumPanel({ item, onPick }: AlbumPanelProps) {
  const payload = item.payload as Record<string, unknown>;
  const artist = (payload.artist as string) ?? item.subtitle ?? "Artista desconhecido";
  const year = (payload.year as number) ?? null;
  const description = (payload.description as string) ?? item.snippet ?? null;
  const tracksCount = (payload.tracks_count as number) ?? null;
  const tags: string[] = Array.isArray(payload.tags) ? (payload.tags as string[]) : [];

  return (
    <div className={styles.panel}>
      <div className={styles.hero}>
        <div className={styles.cover} aria-hidden>
          <span className={styles.coverYear}>{year ?? "—"}</span>
        </div>
        <div className={styles.info}>
          <div className={styles.eyebrow}>Álbum</div>
          <h1 className={styles.title}>{item.title}</h1>
          <div className={styles.byline}>
            por{" "}
            <button
              className={styles.artistLink}
              onClick={() => onPick(artist)}
            >
              {artist}
            </button>
          </div>
          {description ? <p className={styles.desc}>{description}</p> : (
            <p className={styles.descMissing}>Sem descrição disponível.</p>
          )}
          <div className={styles.pills}>
            {tracksCount ? <Tag>{tracksCount} faixas</Tag> : null}
            {tags.map((t) => (
              <Tag key={t}>{t}</Tag>
            ))}
            {tags.length === 0 && !tracksCount ? (
              <span className={styles.placeholder}>Sem metadados adicionais</span>
            ) : null}
          </div>
        </div>
      </div>
      <div className={styles.metaBox}>
        <div className={styles.metaRow}>
          <span>doc_id</span>
          <code>{item.id}</code>
        </div>
        <div className={styles.metaRow}>
          <span>BM25</span>
          <code>{item.score.toFixed(3)}</code>
        </div>
        <div className={styles.metaRow}>
          <span>rank</span>
          <code>#{item.rank}</code>
        </div>
        <div className={styles.metaRow}>
          <span>ano</span>
          <code>{placeholder(year)}</code>
        </div>
      </div>
    </div>
  );
}
