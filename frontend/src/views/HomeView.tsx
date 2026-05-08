import type { ReactNode } from "react";
import { HomeHero } from "../components/home/HomeHero";
import { HomeChips } from "../components/home/HomeChips";
import { HomeStats } from "../components/home/HomeStats";
import styles from "./HomeView.module.css";

interface HomeViewProps {
  onSubmit: (q: string) => void;
  actions?: ReactNode;
}

const SUGGESTIONS = [
  "calcinha preta",
  "garota de ipanema",
  "samba",
  "águas de março",
  "anitta",
  "letra de cobertor",
];

export function HomeView({ onSubmit, actions }: HomeViewProps) {
  return (
    <div className={styles.page}>
      {actions ? <div className={styles.topActions}>{actions}</div> : null}
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
