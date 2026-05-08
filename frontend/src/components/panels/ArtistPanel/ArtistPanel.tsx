import { Tag } from "../../primitives/Tag";
import { ProgressBar } from "../../primitives/ProgressBar";
import { MetaGrid } from "../MetaGrid";
import { TrackList } from "../TrackList";
import { DiscographyGrid } from "../DiscographyGrid";
import type { ArtistResponse } from "../../../api/types";
import { placeholder } from "../../../utils/format";
import styles from "./ArtistPanel.module.css";

interface ArtistPanelProps {
  artist: ArtistResponse;
  onPick: (q: string) => void;
}

export function ArtistPanel({ artist, onPick }: ArtistPanelProps) {
  const heroBg = `linear-gradient(135deg, color-mix(in oklab, var(--accent) 35%, var(--bg)), color-mix(in oklab, var(--accent) 12%, var(--bg)))`;

  const meta = [
    { label: "Origem", value: placeholder(artist.origin) },
    { label: "Início", value: placeholder(artist.year_started) },
    { label: "Ouvintes/mês", value: placeholder(artist.monthly_listeners) },
    {
      label: "Popularidade",
      value:
        artist.popularity != null ? (
          <ProgressBar value={artist.popularity} />
        ) : (
          <span className={styles.placeholder}>indisponível</span>
        ),
    },
  ];

  return (
    <div className={styles.panel}>
      <div className={styles.hero} style={{ background: heroBg }} aria-hidden>
        <span className={styles.heroTag}>artista</span>
      </div>
      <div className={styles.body}>
        <div className={styles.eyebrow}>Knowledge Panel</div>
        <h1 className={styles.title}>{artist.name}</h1>
        {artist.tagline ? (
          <p className={styles.tagline}>{artist.tagline}</p>
        ) : null}
        {artist.bio ? (
          <p className={styles.bio}>{artist.bio}</p>
        ) : (
          <p className={styles.bioMissing}>Sem biografia disponível.</p>
        )}

        <MetaGrid items={meta} columns={2} />

        {artist.genres.length > 0 ? (
          <div className={styles.section}>
            <div className={styles.sectionTitle}>Gêneros</div>
            <div className={styles.pills}>
              {artist.genres.map((g) => (
                <Tag key={g} onClick={() => onPick(g)}>
                  {g}
                </Tag>
              ))}
            </div>
          </div>
        ) : null}

        <div className={styles.section}>
          <div className={styles.sectionTitle}>Top músicas</div>
          <TrackList
            tracks={artist.top_tracks}
            onPick={(t) => onPick(t.title)}
            emptyHint="Sem top tracks disponíveis."
          />
        </div>

        <div className={styles.section}>
          <div className={styles.sectionTitle}>Discografia</div>
          <DiscographyGrid
            albums={artist.albums}
            onPick={(a) => onPick(a.title)}
          />
        </div>

        {artist.source_url ? (
          <div className={styles.footer}>
            <a
              href={artist.source_url}
              target="_blank"
              rel="noreferrer noopener"
              className={styles.sourceLink}
            >
              fonte: {artist.source ?? new URL(artist.source_url).hostname}
            </a>
          </div>
        ) : null}
      </div>
    </div>
  );
}
