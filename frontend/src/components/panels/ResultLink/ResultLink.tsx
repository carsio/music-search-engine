import { useNavigate } from "react-router-dom";
import { Tag } from "../../primitives/Tag";
import type { SearchResultItem } from "../../../api/types";
import styles from "./ResultLink.module.css";

interface ResultLinkProps {
  item: SearchResultItem;
  query?: string;
  onPick: (q: string) => void;
  showScore?: boolean;
}

interface Display {
  site: string;
  path: string;
  faviconAccent: string;
  href?: string;
}

function describe(item: SearchResultItem): Display {
  const payload = item.payload as Record<string, unknown>;
  switch (item.intent) {
    case "track": {
      const album = (payload.album_name as string) ?? "";
      const year = payload.release_year ? ` · ${payload.release_year}` : "";
      return {
        site: "musicabr · letra",
        path: album ? `${album}${year}` : `track/${item.id}`,
        faviconAccent: "var(--tone-track)",
        href: `/song/${item.id}`,
      };
    }
    case "artist": {
      const genres = (payload.genres as string[] | undefined)?.slice(0, 2).join(" · ");
      return {
        site: "musicabr · artista",
        path: genres ?? `artist/${item.id}`,
        faviconAccent: "var(--tone-artist)",
        href: `/artist/${item.id}`,
      };
    }
    case "album":
      return {
        site: "musicabr · álbum",
        path: (payload.artist as string) ?? `album/${item.id}`,
        faviconAccent: "var(--tone-album)",
      };
    case "genre":
      return {
        site: "musicabr · gênero",
        path: (payload.origin as string) ?? `genre/${item.id}`,
        faviconAccent: "var(--tone-genre)",
      };
    case "lyric":
      return {
        site: "musicabr · trecho",
        path: (payload.artist_names as string) ?? `lyric/${item.id}`,
        faviconAccent: "var(--tone-lyric)",
      };
    default:
      return {
        site: "musicabr",
        path: item.id,
        faviconAccent: "var(--bg-2)",
      };
  }
}

export function ResultLink({ item, query, onPick, showScore = true }: ResultLinkProps) {
  const desc = describe(item);
  const navigate = useNavigate();

  function handleActivate() {
    if (desc.href) {
      const search = query ? `?q=${encodeURIComponent(query)}` : "";
      navigate(`${desc.href}${search}`);
    } else {
      onPick(item.title);
    }
  }

  return (
    <article
      className={styles.result}
      onClick={handleActivate}
      role="link"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleActivate();
        }
      }}
    >
      <div className={styles.meta}>
        <span
          className={styles.favicon}
          style={{ background: desc.faviconAccent }}
          aria-hidden
        />
        <span className={styles.urlStack}>
          <span className={styles.site}>{desc.site}</span>
          <span className={styles.path}>{desc.path}</span>
        </span>
        {showScore ? (
          <Tag variant="score">
            #{item.rank} · {item.score.toFixed(2)}
          </Tag>
        ) : null}
      </div>
      <h3 className={styles.title}>{item.title}</h3>
      {item.subtitle ? <div className={styles.subtitle}>{item.subtitle}</div> : null}
      {item.snippet ? <p className={styles.snippet}>{item.snippet}</p> : null}
    </article>
  );
}
