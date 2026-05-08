import type { AlbumRef } from "../../../api/types";
import styles from "./DiscographyGrid.module.css";

interface DiscographyGridProps {
  albums: AlbumRef[];
  onPick?: (album: AlbumRef) => void;
}

export function DiscographyGrid({ albums, onPick }: DiscographyGridProps) {
  if (albums.length === 0) {
    return <div className={styles.empty}>Sem discografia disponível.</div>;
  }
  return (
    <div className={styles.grid}>
      {albums.map((album, i) => (
        <button
          key={`${album.title}-${i}`}
          className={styles.card}
          onClick={() => onPick?.(album)}
        >
          <div
            className={styles.cover}
            style={{
              background: `linear-gradient(135deg, color-mix(in oklab, var(--accent) 30%, var(--bg)), var(--bg-2))`,
            }}
            aria-hidden
          >
            {album.year ? <span className={styles.year}>{album.year}</span> : null}
          </div>
          <div className={styles.title}>{album.title}</div>
          {album.tracks ? (
            <div className={styles.tracks}>{album.tracks} faixas</div>
          ) : null}
        </button>
      ))}
    </div>
  );
}
