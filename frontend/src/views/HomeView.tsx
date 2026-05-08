import { HomeHero } from "../components/home/HomeHero";
import { HomeChips } from "../components/home/HomeChips";
import { HomeStats } from "../components/home/HomeStats";

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
    <>
      <HomeHero onSubmit={onSubmit} />
      <HomeChips items={SUGGESTIONS} onPick={onSubmit} />
      <HomeStats />
    </>
  );
}
