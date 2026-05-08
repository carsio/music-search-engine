import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { api } from "./client";
import type {
  ArtistResponse,
  HealthResponse,
  LyricSearchResponse,
  SearchAlgorithm,
  SearchResponse,
  SongResponse,
} from "./types";

type Opts<T> = Pick<UseQueryOptions<T>, "enabled" | "staleTime">;

interface SearchOpts extends Opts<SearchResponse> {
  algorithm?: SearchAlgorithm;
  rerank?: boolean;
  top?: number;
}

export function useSearch(query: string, opts: SearchOpts = {}) {
  const { algorithm = "bm25", rerank = false, top = 10, enabled, staleTime = 60_000 } = opts;
  return useQuery({
    enabled: (enabled ?? true) && query.trim().length > 0,
    staleTime,
    placeholderData: (prev) => prev,
    queryKey: ["search", query, algorithm, rerank, top],
    queryFn: async () => {
      const { data } = await api.get<SearchResponse>("/search", {
        params: { q: query, top, algorithm, rerank },
      });
      return data;
    },
  });
}

interface LyricOpts extends Opts<LyricSearchResponse> {
  algorithm?: SearchAlgorithm;
  top?: number;
}

export function useLyricSearch(query: string, opts: LyricOpts = {}) {
  const { algorithm = "bm25", top = 20, enabled, staleTime = 60_000 } = opts;
  return useQuery({
    enabled: (enabled ?? true) && query.trim().length > 0,
    staleTime,
    placeholderData: (prev) => prev,
    queryKey: ["lyric", query, algorithm, top],
    queryFn: async () => {
      const { data } = await api.get<LyricSearchResponse>("/search/lyric", {
        params: { q: query, top, algorithm },
      });
      return data;
    },
  });
}

export function useArtist(
  id: string | null | undefined,
  opts: Opts<ArtistResponse> = {},
) {
  const { enabled, staleTime = 60_000 } = opts;
  return useQuery({
    enabled: (enabled ?? true) && !!id,
    staleTime,
    queryKey: ["artist", id],
    queryFn: async () => {
      const { data } = await api.get<ArtistResponse>(`/artist/${id}`);
      return data;
    },
  });
}

export function useSong(id: string | null | undefined, opts: Opts<SongResponse> = {}) {
  const { enabled, staleTime = 60_000 } = opts;
  return useQuery({
    enabled: (enabled ?? true) && !!id,
    staleTime,
    queryKey: ["song", id],
    queryFn: async () => {
      const { data } = await api.get<SongResponse>(`/song/${id}`);
      return data;
    },
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    staleTime: 5 * 60_000,
    queryFn: async () => {
      const { data } = await api.get<HealthResponse>("/healthz");
      return data;
    },
  });
}
