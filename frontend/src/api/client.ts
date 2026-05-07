import axios from "axios";
import { useQuery } from "@tanstack/react-query";
import type {
  ArtistResponse,
  LyricSearchResponse,
  SearchAlgorithm,
  SearchResponse,
  SongResponse,
} from "../types";

const api = axios.create({ baseURL: "/api", timeout: 30_000 });

export function useSearch(query: string, opts?: { algorithm?: SearchAlgorithm; rerank?: boolean }) {
  return useQuery({
    enabled: query.trim().length > 0,
    queryKey: ["search", query, opts?.algorithm ?? "bm25", opts?.rerank ?? false],
    queryFn: async () => {
      const { data } = await api.get<SearchResponse>("/search", {
        params: {
          q: query,
          top: 10,
          algorithm: opts?.algorithm ?? "bm25",
          rerank: opts?.rerank ?? false,
        },
      });
      return data;
    },
  });
}

export function useLyricSearch(query: string, algorithm: SearchAlgorithm = "bm25") {
  return useQuery({
    enabled: query.trim().length > 0,
    queryKey: ["lyric", query, algorithm],
    queryFn: async () => {
      const { data } = await api.get<LyricSearchResponse>("/search/lyric", {
        params: { q: query, top: 20, algorithm },
      });
      return data;
    },
  });
}

export function useArtist(id: string | null) {
  return useQuery({
    enabled: !!id,
    queryKey: ["artist", id],
    queryFn: async () => {
      const { data } = await api.get<ArtistResponse>(`/artist/${id}`);
      return data;
    },
  });
}

export function useSong(id: string | null) {
  return useQuery({
    enabled: !!id,
    queryKey: ["song", id],
    queryFn: async () => {
      const { data } = await api.get<SongResponse>(`/song/${id}`);
      return data;
    },
  });
}

export async function fetchHealth() {
  const { data } = await api.get("/healthz");
  return data;
}
