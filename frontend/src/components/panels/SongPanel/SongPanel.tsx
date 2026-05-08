import { MetaBar, type MetaBarItem } from "../MetaBar";
import { LyricsBlock } from "../LyricsBlock";
import type { SongResponse } from "../../../api/types";
import { placeholder } from "../../../utils/format";
import styles from "./SongPanel.module.css";

interface SongPanelProps {
  song: SongResponse;
  query: string;
}

export function SongPanel({ song, query }: SongPanelProps) {
  const meta: MetaBarItem[] = [
    { label: "Álbum", value: placeholder(song.album) },
    { label: "Ano", value: placeholder(song.year) },
    { label: "Duração", value: placeholder(song.duration) },
    { label: "Plays", value: placeholder(song.plays) },
    {
      label: "Compositor",
      value: song.composers.length > 0 ? song.composers[0] : "—",
    },
  ];

  const headerBg = `linear-gradient(135deg, color-mix(in oklab, var(--accent) 30%, var(--bg)), color-mix(in oklab, var(--accent) 8%, var(--bg)))`;

  return (
    <article className={styles.panel}>
      <header className={styles.header} style={{ background: headerBg }}>
        <div className={styles.eyebrow}>Música</div>
        <h1 className={styles.title}>{song.title}</h1>
        <div className={styles.byline}>
          por <strong>{song.artist}</strong>
        </div>
      </header>
      <MetaBar items={meta} />
      <LyricsBlock lyrics={song.lyrics} query={query} />
      <footer className={styles.footer}>
        <span className={styles.tag}>doc_id: {song.id}</span>
        {song.macro_genre ? (
          <span className={styles.tag}>macro: {song.macro_genre}</span>
        ) : null}
        {song.lyrics_source ? (
          <span className={styles.tag}>fonte: {song.lyrics_source}</span>
        ) : null}
        {song.lyrics_source_url ? (
          <a
            href={song.lyrics_source_url}
            target="_blank"
            rel="noreferrer noopener"
            className={styles.tagLink}
          >
            ver letra original ↗
          </a>
        ) : null}
      </footer>
    </article>
  );
}
