import type { SearchResultItem } from "../../../api/types";
import styles from "./GenrePanel.module.css";

interface GenrePanelProps {
  top?: SearchResultItem;
  related: SearchResultItem[];
  onPick: (q: string) => void;
}

export function GenrePanel({ top, related, onPick }: GenrePanelProps) {
  if (!top) {
    return null;
  }
  const payload = top.payload as Record<string, unknown>;
  const description = (payload.description as string) ?? top.snippet ?? null;
  const origin = (payload.origin as string) ?? top.subtitle ?? null;
  const eraStart = (payload.era_start as number | string) ?? null;

  return (
    <div className={styles.wrap}>
      <section className={styles.hero}>
        <div className={styles.eyebrow}>Gênero</div>
        <h1 className={styles.title}>{top.title}</h1>
        {description ? (
          <p className={styles.desc}>{description}</p>
        ) : (
          <p className={styles.descMissing}>Sem descrição disponível para este gênero.</p>
        )}
        <div className={styles.stats}>
          {origin ? (
            <div>
              <strong>{origin}</strong>
              <span>origem</span>
            </div>
          ) : null}
          {eraStart ? (
            <div>
              <strong>{eraStart}</strong>
              <span>desde</span>
            </div>
          ) : null}
          <div>
            <strong>{related.length}</strong>
            <span>resultados</span>
          </div>
        </div>
      </section>

      {related.length > 0 ? (
        <div className={styles.grid}>
          {related.map((item) => (
            <button
              key={item.id}
              className={styles.card}
              onClick={() => onPick(item.title)}
            >
              <div className={styles.cardCover} aria-hidden />
              <div className={styles.cardInfo}>
                <div className={styles.cardName}>{item.title}</div>
                {item.subtitle ? (
                  <div className={styles.cardMeta}>{item.subtitle}</div>
                ) : null}
              </div>
            </button>
          ))}
        </div>
      ) : (
        <div className={styles.empty}>Sem artistas relacionados nesta busca.</div>
      )}
    </div>
  );
}
