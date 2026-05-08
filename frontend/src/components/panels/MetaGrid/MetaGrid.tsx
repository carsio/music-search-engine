import type { ReactNode } from "react";
import styles from "./MetaGrid.module.css";

export interface MetaItem {
  label: string;
  value: ReactNode;
}

interface MetaGridProps {
  items: MetaItem[];
  columns?: number;
}

export function MetaGrid({ items, columns = 2 }: MetaGridProps) {
  return (
    <div
      className={styles.grid}
      style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}
    >
      {items.map((item) => (
        <div key={item.label}>
          <div className={styles.label}>{item.label}</div>
          <div className={styles.value}>{item.value}</div>
        </div>
      ))}
    </div>
  );
}
