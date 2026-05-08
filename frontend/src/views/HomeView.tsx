import { HomeHero } from "../components/home/HomeHero";
import { HomeChips } from "../components/home/HomeChips";
import { HomeStats } from "../components/home/HomeStats";
import styles from "./HomeView.module.css";

interface HomeViewProps {
  onSubmit: (q: string) => void;
}

const SUGGESTIONS = [
  "calcinha preta",
  "garota de ipanema",
  "samba",
  "águas de março",
  "anitta",
  "letra de cobertor",
];

export function HomeView({ onSubmit }: HomeViewProps) {
  return (
    <div className={styles.page}>
      <div className={styles.heroBlock}>
        <HomeHero onSubmit={onSubmit} />
      </div>
      <div className={styles.meta}>
        <HomeChips items={SUGGESTIONS} onPick={onSubmit} />
        <HomeStats />
      </div>
    </div>
  );
}
