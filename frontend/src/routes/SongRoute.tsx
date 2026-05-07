import { useParams, useSearchParams } from "react-router-dom";
import { useSong } from "../api/client";
import { SongPanel } from "../components/SongPanel";

export function SongRoute() {
  const { id } = useParams<{ id: string }>();
  const [params] = useSearchParams();
  const query = params.get("q") ?? undefined;
  const { data, isPending, isError, error } = useSong(id ?? null);
  if (isPending) return <div className="loading">Carregando música...</div>;
  if (isError) return <div className="error">Erro: {String(error)}</div>;
  if (!data) return null;
  return <SongPanel song={data} query={query} />;
}
