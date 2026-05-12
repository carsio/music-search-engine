import { useNavigate } from "react-router-dom";
import { useQueryParam } from "../../../hooks/useQueryParam";
import { useSearchSettings } from "../../../hooks/useSearchSettings";
import { useHealth } from "../../../api/hooks";
import { SearchHeader } from "../../search/SearchHeader";
import { SearchSettingsMenu } from "../../search/SearchSettingsMenu";
import { HomeView } from "../../../views/HomeView";
import { ResultsRouter } from "../../../views/ResultsRouter";
import { formatNumber } from "../../../utils/format";
import styles from "./AppShell.module.css";

export function AppShell() {
  const [query, setQuery] = useQueryParam("q");
  const { settings, hasActiveOverrides, updateSettings, resetSettings } = useSearchSettings();
  const navigate = useNavigate();
  const health = useHealth();

  const stats = health.data
    ? [
        { label: "letras", value: formatNumber(health.data.tracks_indexed) },
        ...Object.entries(health.data.entities).map(([k, v]) => ({
          label: k,
          value: formatNumber(v),
        })),
      ]
    : undefined;

  const hasQuery = query.trim().length > 0;
  const settingsMenu = (
    <SearchSettingsMenu
      settings={settings}
      hasActiveOverrides={hasActiveOverrides}
      onUpdate={updateSettings}
      onReset={resetSettings}
    />
  );

  return (
    <div className={styles.app}>
      {hasQuery ? (
        <SearchHeader
          query={query}
          onSubmit={setQuery}
          onLogoClick={() => navigate("/")}
          stats={stats}
          actions={settingsMenu}
        />
      ) : null}
      <main className={styles.main}>
        {hasQuery ? (
          <ResultsRouter query={query} onSubmit={setQuery} settings={settings} />
        ) : (
          <HomeView onSubmit={setQuery} actions={settingsMenu} />
        )}
      </main>
    </div>
  );
}
