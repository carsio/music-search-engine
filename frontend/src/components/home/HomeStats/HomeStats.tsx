import { useHealth } from "../../../api/hooks";
import { formatNumber } from "../../../utils/format";
import styles from "./HomeStats.module.css";

interface StatCell {
  label: string;
  value: string;
}

export function HomeStats() {
  const { data, isLoading } = useHealth();

  let cells: StatCell[];
  if (isLoading || !data) {
    cells = [
      { label: "letras", value: "—" },
      { label: "artistas", value: "—" },
      { label: "álbuns", value: "—" },
      { label: "llm", value: "—" },
    ];
  } else {
    const ents = data.entities ?? {};
    cells = [
      { label: "letras", value: formatNumber(data.tracks_indexed) },
      { label: "artistas", value: formatNumber(ents.artist ?? 0) },
      { label: "álbuns", value: formatNumber(ents.album ?? 0) },
      { label: "llm", value: data.llm ? "ativa" : "off" },
    ];
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.grid}>
        {cells.map((cell) => (
          <div key={cell.label} className={styles.cell}>
            <div className={styles.num}>{cell.value}</div>
            <div className={styles.label}>{cell.label}</div>
          </div>
        ))}
      </div>
      <p className={styles.foot}>
        UFAM · ICC222 · 2026.1 · BM25 + TF-IDF + LLM rerank opcional
      </p>
    </div>
  );
}
