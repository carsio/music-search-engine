import { useParams } from "react-router-dom";
import { useArtist } from "../api/client";
import { ArtistPanel } from "../components/ArtistPanel";

export function ArtistRoute() {
  const { id } = useParams<{ id: string }>();
  const { data, isPending, isError, error } = useArtist(id ?? null);
  if (isPending) return <div className="loading">Carregando artista...</div>;
  if (isError) return <div className="error">Erro: {String(error)}</div>;
  if (!data) return null;
  return <ArtistPanel artist={data} />;
}
