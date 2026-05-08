import { ResultsLayout } from "../components/layout/ResultsLayout";
import { EmptyState } from "../components/states/EmptyState";

interface NoResultsViewProps {
  query: string;
  onSubmit: (q: string) => void;
}

const SUGGESTIONS = ["caetano veloso", "anitta", "samba", "águas de março"];

export function NoResultsView({ query, onSubmit }: NoResultsViewProps) {
  return (
    <ResultsLayout
      variant="one-col"
      eyebrow={
        <>
          nenhum resultado para <strong>{query}</strong>
        </>
      }
      primary={
        <EmptyState
          title="Não encontramos nada por aqui."
          hint="Tente termos mais gerais, verifique a ortografia, ou comece com um artista, música ou gênero conhecido."
          suggestions={SUGGESTIONS}
          onPick={onSubmit}
        />
      }
    />
  );
}
