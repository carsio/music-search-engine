import { useEffect, useRef, useState } from "react";
import { Button } from "../../primitives/Button";
import { Card } from "../../primitives/Card";
import { Chip } from "../../primitives/Chip";
import { Icon } from "../../primitives/Icon";
import type {
  SearchProfile,
  SearchSettings,
  SearchTfScheme,
} from "../../../hooks/useSearchSettings";
import styles from "./SearchSettingsMenu.module.css";

interface SearchSettingsMenuProps {
  settings: SearchSettings;
  hasActiveOverrides: boolean;
  onUpdate: (patch: Partial<SearchSettings>) => void;
  onReset: () => void;
}

const PROFILE_COPY: Record<SearchProfile, { title: string; sub: string }> = {
  balanced: {
    title: "Balanceado",
    sub: "Mistura campo textual, metadados e roteamento padrão.",
  },
  lyrics: {
    title: "Foco em letra",
    sub: "Puxa mais peso para trechos da música e snippets.",
  },
  metadata: {
    title: "Foco em metadados",
    sub: "Favorece título, artista, álbum e contexto semântico.",
  },
};

const TF_SCHEME_COPY: Record<SearchTfScheme, string> = {
  raw: "Bruto",
  log: "Log",
  augmented: "Augmented",
};

function formatAlgorithmLabel(settings: SearchSettings) {
  return settings.algorithm === "bm25" ? "BM25" : "TF-IDF";
}

export function SearchSettingsMenu({
  settings,
  hasActiveOverrides,
  onUpdate,
  onReset,
}: SearchSettingsMenuProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }

    function handlePointerDown(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);

    // Lock body scroll only when the panel is rendered as a full-screen
    // sheet on small screens — avoids the awkward double-scroll on iOS.
    const isMobileSheet =
      typeof window !== "undefined" && window.matchMedia("(max-width: 720px)").matches;
    const previousOverflow = document.body.style.overflow;
    if (isMobileSheet) {
      document.body.style.overflow = "hidden";
    }
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  return (
    <div className={styles.wrap} ref={rootRef}>
      <button
        type="button"
        className={styles.trigger}
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label="Abrir configurações da busca"
      >
        <span className={styles.triggerIcon} aria-hidden>
          <Icon name="settings" size={16} />
        </span>
        <span className={styles.triggerCopy}>
          <span className={styles.triggerLabel}>Busca</span>
          <span className={styles.triggerMeta}>
            {formatAlgorithmLabel(settings)} · {PROFILE_COPY[settings.profile].title}
          </span>
        </span>
        {hasActiveOverrides ? <span className={styles.triggerDot} aria-hidden /> : null}
      </button>

      {open ? (
        <Card
          variant="raised"
          padding="sm"
          className={styles.panel}
          role="dialog"
          aria-label="Configurações da busca"
        >
          <div className={styles.panelHeader}>
            <div>
              <p className={styles.eyebrow}>Motor de recuperação</p>
              <h2 className={styles.title}>Ajuste a forma como a busca acontece</h2>
            </div>
            <Button variant="ghost" size="sm" icon={<Icon name="close" size={16} />} onClick={() => setOpen(false)} aria-label="Fechar configurações" />
          </div>

          <section className={styles.section}>
            <div className={styles.sectionHead}>
              <h3 className={styles.sectionTitle}>Básico</h3>
              <span className={styles.sectionMeta}>{settings.top} resultados</span>
            </div>
            <div className={styles.inlineOptions}>
              <Chip
                active={settings.algorithm === "bm25"}
                onClick={() => onUpdate({ algorithm: "bm25" })}
              >
                BM25
              </Chip>
              <Chip
                active={settings.algorithm === "tfidf"}
                onClick={() => onUpdate({ algorithm: "tfidf" })}
              >
                TF-IDF
              </Chip>
            </div>

            <label className={styles.rangeField}>
              <span className={styles.rangeHead}>
                <span className={styles.rangeLabel}>Quantidade de resultados</span>
                <span className={styles.rangeValue}>{settings.top}</span>
              </span>
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={settings.top}
                onChange={(event) => onUpdate({ top: Number(event.target.value) })}
              />
            </label>
          </section>

          <section className={styles.section}>
            <div className={styles.sectionHead}>
              <h3 className={styles.sectionTitle}>Foco</h3>
              <span className={styles.sectionMeta}>{PROFILE_COPY[settings.profile].title}</span>
            </div>
            <div className={styles.profileGrid}>
              {(Object.entries(PROFILE_COPY) as [SearchProfile, (typeof PROFILE_COPY)[SearchProfile]][]).map(
                ([profile, copy]) => (
                  <Chip
                    key={profile}
                    variant="related"
                    active={settings.profile === profile}
                    eyebrow="preset"
                    sub={copy.sub}
                    onClick={() => onUpdate({ profile })}
                  >
                    {copy.title}
                  </Chip>
                ),
              )}
            </div>
          </section>

          <section className={styles.section}>
            <div className={styles.sectionHead}>
              <h3 className={styles.sectionTitle}>Avançado</h3>
              <span className={styles.sectionMeta}>{formatAlgorithmLabel(settings)}</span>
            </div>

            {settings.algorithm === "bm25" ? (
              <div className={styles.advancedGrid}>
                <label className={styles.rangeField}>
                  <span className={styles.rangeHead}>
                    <span className={styles.rangeLabel}>BM25 k1</span>
                    <span className={styles.rangeValue}>{settings.bm25K1.toFixed(1)}</span>
                  </span>
                  <input
                    type="range"
                    min={0.1}
                    max={3}
                    step={0.1}
                    value={settings.bm25K1}
                    onChange={(event) => onUpdate({ bm25K1: Number(event.target.value) })}
                  />
                </label>

                <label className={styles.rangeField}>
                  <span className={styles.rangeHead}>
                    <span className={styles.rangeLabel}>BM25 b</span>
                    <span className={styles.rangeValue}>{settings.bm25B.toFixed(2)}</span>
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={settings.bm25B}
                    onChange={(event) => onUpdate({ bm25B: Number(event.target.value) })}
                  />
                </label>
              </div>
            ) : (
              <div className={styles.inlineOptions}>
                {(Object.entries(TF_SCHEME_COPY) as [SearchTfScheme, string][]).map(
                  ([tfScheme, label]) => (
                    <Chip
                      key={tfScheme}
                      active={settings.tfScheme === tfScheme}
                      onClick={() => onUpdate({ tfScheme })}
                    >
                      {label}
                    </Chip>
                  ),
                )}
              </div>
            )}
          </section>

          <section className={styles.section}>
            <div className={styles.sectionHead}>
              <h3 className={styles.sectionTitle}>Trechos de letra</h3>
              <span className={styles.sectionMeta}>{settings.maxSnippets} snippets</span>
            </div>
            <label className={styles.rangeField}>
              <span className={styles.rangeHead}>
                <span className={styles.rangeLabel}>Quantidade máxima de snippets</span>
                <span className={styles.rangeValue}>{settings.maxSnippets}</span>
              </span>
              <input
                type="range"
                min={1}
                max={10}
                step={1}
                value={settings.maxSnippets}
                onChange={(event) => onUpdate({ maxSnippets: Number(event.target.value) })}
              />
            </label>
          </section>

          <div className={styles.footer}>
            <p className={styles.footerNote}>
              As configurações ficam salvas na sessão e também entram na URL da busca.
            </p>
            <Button variant="outline" size="sm" onClick={onReset}>
              Resetar padrão
            </Button>
          </div>
        </Card>
      ) : null}
    </div>
  );
}