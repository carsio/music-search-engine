import { ResultsLayout } from "../components/layout/ResultsLayout";
import { AlbumPanel } from "../components/panels/AlbumPanel";
import { ResultsList } from "../components/panels/ResultsList";
import { Card } from "../components/primitives/Card";
import { Tag } from "../components/primitives/Tag";
import { Section } from "../components/layout/Section";
import type { SearchResponse } from "../api/types";

interface AlbumViewProps {
  query: string;
  response: SearchResponse;
  onSubmit: (q: string) => void;
}

export function AlbumView({ query, response, onSubmit }: AlbumViewProps) {
  const top = response.items[0];
  const items = response.items;

  const main = (
    <>
      <Card variant="dashed" padding="sm" style={{ marginBottom: 24 }}>
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 10,
            color: "var(--ink-3)",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            marginBottom: 8,
          }}
        >
          Por que esse resultado?
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          <Tag variant="ri">
            match em <code>album.title</code>
          </Tag>
          <Tag variant="ri">
            algoritmo: <code>{response.algorithm}</code>
          </Tag>
          <Tag variant="ri">
            latência: <code>{response.elapsed_ms}ms</code>
          </Tag>
        </div>
      </Card>

      <Section
        title="Resultados"
        meta={<>{items.length} resultados · BM25</>}
      >
        <ResultsList items={items} onPick={onSubmit} />
      </Section>
    </>
  );

  return (
    <ResultsLayout
      eyebrow={
        <>
          Resultados para <strong>{query}</strong>
        </>
      }
      primary={main}
      secondary={top ? <AlbumPanel item={top} onPick={onSubmit} /> : undefined}
    />
  );
}
