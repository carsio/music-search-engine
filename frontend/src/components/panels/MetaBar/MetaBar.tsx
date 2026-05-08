import type { ReactNode } from "react";
import styles from "./MetaBar.module.css";

export interface MetaBarItem {
  label: string;
  value: ReactNode;
}

interface MetaBarProps {
  items: MetaBarItem[];
}

export function MetaBar({ items }: MetaBarProps) {
  return (
    <div
      className={styles.bar}
      style={{ gridTemplateColumns: `repeat(${items.length}, 1fr)` }}
    >
      {items.map((item) => (
        <div key={item.label} className={styles.cell}>
          <div className={styles.label}>{item.label}</div>
          <div className={styles.value}>{item.value}</div>
        </div>
      ))}
    </div>
  );
}
