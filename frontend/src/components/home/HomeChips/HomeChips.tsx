import { Chip } from "../../primitives/Chip";
import styles from "./HomeChips.module.css";

interface HomeChipsProps {
  items: string[];
  onPick: (q: string) => void;
}

export function HomeChips({ items, onPick }: HomeChipsProps) {
  return (
    <div className={styles.row}>
      <span className={styles.label}>Tente</span>
      {items.map((item) => (
        <Chip key={item} variant="quick" onClick={() => onPick(item)}>
          {item}
        </Chip>
      ))}
    </div>
  );
}
