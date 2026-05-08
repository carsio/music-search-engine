import type { TrackRef } from "../../../api/types";
import { cx } from "../../../utils/classnames";
import styles from "./TrackList.module.css";

interface TrackListProps {
  tracks: TrackRef[];
  onPick?: (track: TrackRef) => void;
  variant?: "default" | "mini";
  emptyHint?: string;
}

export function TrackList({
  tracks,
  onPick,
  variant = "default",
  emptyHint = "Sem faixas disponíveis.",
}: TrackListProps) {
  if (tracks.length === 0) {
    return <div className={styles.empty}>{emptyHint}</div>;
  }
  return (
    <ol className={cx(styles.list, variant === "mini" && styles.mini)}>
      {tracks.map((track, i) => (
        <li
          key={`${track.title}-${i}`}
          className={styles.track}
          onClick={() => onPick?.(track)}
          role={onPick ? "button" : undefined}
          tabIndex={onPick ? 0 : undefined}
          onKeyDown={(e) => {
            if (onPick && (e.key === "Enter" || e.key === " ")) {
              e.preventDefault();
              onPick(track);
            }
          }}
        >
          <span className={styles.num}>{String(i + 1).padStart(2, "0")}</span>
          <span className={styles.title}>{track.title}</span>
          {track.album ? (
            <span className={styles.album}>{track.album}</span>
          ) : (
            <span className={styles.album} />
          )}
          {variant === "default" && track.plays ? (
            <span className={styles.plays}>{track.plays}</span>
          ) : null}
        </li>
      ))}
    </ol>
  );
}
