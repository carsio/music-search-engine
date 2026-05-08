import { Card } from "../../primitives/Card";
import { Tag } from "../../primitives/Tag";
import { DiscographyGrid } from "../DiscographyGrid";
import { TrackList } from "../TrackList";
import type { AlbumResponse, SearchResultItem } from "../../../api/types";
import { placeholder } from "../../../utils/format";
import styles from "./AlbumPanel.module.css";

interface AlbumPanelProps {
  album: AlbumResponse;
  item: SearchResultItem;
  onPick: (q: string) => void;
}

interface AlbumSidebarProps {
  album: AlbumResponse;
  item: SearchResultItem;
  onPick: (q: string) => void;
}

export function AlbumPanel({ album, item, onPick }: AlbumPanelProps) {
  const artist = album.artist || item.subtitle || "Artista desconhecido";
  const year = album.year ?? null;
  const description = album.description ?? item.snippet ?? null;
  const coverStyle = album.cover_url
    ? {
        backgroundImage: `linear-gradient(180deg, rgba(0, 0, 0, 0.08), rgba(0, 0, 0, 0.62)), url(${album.cover_url})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
      }
    : undefined;

  return (
    <article className={styles.panel}>
      <div className={styles.hero}>
        <div className={styles.cover} style={coverStyle} aria-hidden>
          <div className={styles.coverOverlay}>
            <span className={styles.coverArtist}>{artist.toUpperCase()}</span>
            <span className={styles.coverTitle}>{album.title}</span>
            <span className={styles.coverYear}>{year ?? "—"}</span>
          </div>
        </div>
        <div className={styles.info}>
          <div className={styles.eyebrow}>Álbum</div>
          <h1 className={styles.title}>{album.title}</h1>
          <div className={styles.byline}>
            <button className={styles.artistLink} onClick={() => onPick(artist)}>
              {artist}
            </button>
            <span>· Álbum</span>
            {year ? <span>· {year}</span> : null}
            {album.tracks_count ? <span>· {album.tracks_count} faixas</span> : null}
          </div>
          {description ? <p className={styles.desc}>{description}</p> : (
            <p className={styles.descMissing}>Sem descrição disponível.</p>
          )}
          <div className={styles.pills}>
            {album.album_type ? <Tag>{album.album_type}</Tag> : null}
            {album.label ? <Tag>{album.label}</Tag> : null}
            {album.duration ? <Tag>{album.duration}</Tag> : null}
            {album.tags.map((tag) => (
              <Tag key={tag} onClick={() => onPick(tag)}>{tag}</Tag>
            ))}
            {album.tags.length === 0 && !album.label && !album.album_type ? (
              <span className={styles.placeholder}>Sem metadados adicionais</span>
            ) : null}
          </div>
        </div>
      </div>

      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <h2 className={styles.sectionTitle}>Faixas</h2>
          <span className={styles.sectionMeta}>
            {album.tracks.length > 0 ? `${album.tracks.length} listadas` : "sem tracklist detalhada"}
          </span>
        </div>
        {album.tracks.length > 0 ? (
          <ol className={styles.trackList}>
            {album.tracks.map((track, index) => (
              <li
                key={track.id}
                className={styles.trackRow}
                onClick={() => onPick(track.title)}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onPick(track.title);
                  }
                }}
              >
                <span className={styles.trackNum}>
                  {String(track.track_number ?? index + 1).padStart(2, "0")}
                </span>
                <span className={styles.trackTitleWrap}>
                  <span className={styles.trackTitle}>{track.title}</span>
                  <span className={styles.trackSub}>
                    {track.disc_number && track.disc_number > 1 ? `Disco ${track.disc_number}` : "Faixa do álbum"}
                    {track.explicit ? " · explícita" : ""}
                  </span>
                </span>
                <span className={styles.trackPlays}>
                  {track.popularity != null ? `pop ${track.popularity}` : "—"}
                </span>
                <span className={styles.trackDuration}>{track.duration ?? "—"}</span>
              </li>
            ))}
          </ol>
        ) : (
          <div className={styles.trackFallback}>A busca encontrou o álbum, mas a tracklist detalhada não está disponível.</div>
        )}
      </section>

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
    </article>
  );
}

export function AlbumSidebar({ album, item, onPick }: AlbumSidebarProps) {
  const artist = album.artist_summary;
  const artistName = artist.name || album.artist;
  const heroStyle = artist.image_url
    ? {
        backgroundImage: `linear-gradient(180deg, rgba(0, 0, 0, 0.12), rgba(0, 0, 0, 0.72)), url(${artist.image_url})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
      }
    : undefined;
  const artistGenres = artist.genres.length > 0 ? artist.genres : album.tags;

  return (
    <div className={styles.sidebar}>
      <Card variant="flat" padding="md" className={styles.sidebarCard}>
        <div className={styles.artistHero} style={heroStyle} aria-hidden>
          <span className={styles.artistHeroLabel}>Sobre o artista</span>
        </div>
        <h3 className={styles.artistName}>{artistName}</h3>
        <p className={styles.artistMeta}>
          {artist.followers_total != null
            ? `${new Intl.NumberFormat("pt-BR").format(artist.followers_total)} seguidores`
            : "Resumo derivado do dataset atual"}
          {artist.popularity != null ? ` · pop ${artist.popularity}` : ""}
        </p>
        <div className={styles.pills}>
          {artistGenres.map((genre) => (
            <button
              key={genre}
              className={styles.sidebarTagButton}
              onClick={() => onPick(genre)}
            >
              <Tag>{genre}</Tag>
            </button>
          ))}
        </div>
        <button className={styles.artistCta} onClick={() => onPick(artistName)}>
          Página de {artistName} →
        </button>
      </Card>

      <Card variant="dashed" padding="md" className={styles.sidebarCard}>
        <div className={styles.sidebarSectionTitle}>Top faixas</div>
        <TrackList
          tracks={artist.top_tracks}
          onPick={(track) => onPick(track.title)}
          variant="mini"
          emptyHint="Sem top faixas disponíveis."
        />
      </Card>

      <Card variant="dashed" padding="md" className={styles.sidebarCard}>
        <div className={styles.sidebarSectionTitle}>Discografia</div>
        <DiscographyGrid
          albums={artist.albums}
          onPick={(albumRef) => onPick(albumRef.title)}
        />
      </Card>

      <Card variant="flat" padding="md" className={styles.sidebarCard}>
        <div className={styles.sidebarSectionTitle}>Metadados RI</div>
        <div className={styles.metaRow}>
          <span>tipo</span>
          <code>album</code>
        </div>
        <div className={styles.metaRow}>
          <span>faixas</span>
          <code>{placeholder(album.tracks_count)}</code>
        </div>
        <div className={styles.metaRow}>
          <span>duração</span>
          <code>{placeholder(album.duration)}</code>
        </div>
        <div className={styles.metaRow}>
          <span>rank</span>
          <code>#{item.rank}</code>
        </div>
      </Card>
    </div>
  );
}
