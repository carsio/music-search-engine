import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { api } from "./client";
import type {
  AlbumResponse,
  ArtistResponse,
  HealthResponse,
  LyricSearchResponse,
  SearchAlgorithm,
  SearchResponse,
  SongResponse,
} from "./types";
import type { SearchProfile, SearchTfScheme } from "../hooks/useSearchSettings";

type Opts<T> = Pick<UseQueryOptions<T>, "enabled" | "staleTime">;

interface SearchOpts extends Opts<SearchResponse> {
  algorithm?: SearchAlgorithm;
  top?: number;
  profile?: SearchProfile;
  bm25K1?: number;
  bm25B?: number;
  tfScheme?: SearchTfScheme;
}

export function useSearch(query: string, opts: SearchOpts = {}) {
  const {
    algorithm = "bm25",
    top = 10,
    profile = "balanced",
    bm25K1 = 1.5,
    bm25B = 0.75,
    tfScheme = "log",
    enabled,
    staleTime = 60_000,
  } = opts;
  return useQuery({
    enabled: (enabled ?? true) && query.trim().length > 0,
    staleTime,
    placeholderData: (prev) => prev,
    queryKey: ["search", query, algorithm, top, profile, bm25K1, bm25B, tfScheme],
    queryFn: async () => {
      const { data } = await api.get<SearchResponse>("/search", {
        params: {
          q: query,
          top,
          algorithm,
          profile,
          bm25_k1: bm25K1,
          bm25_b: bm25B,
          tf_scheme: tfScheme,
        },
      });
      return data;
    },
  });
}

interface LyricOpts extends Opts<LyricSearchResponse> {
  algorithm?: SearchAlgorithm;
  top?: number;
  profile?: SearchProfile;
  bm25K1?: number;
  bm25B?: number;
  tfScheme?: SearchTfScheme;
  maxSnippets?: number;
}

export function useLyricSearch(query: string, opts: LyricOpts = {}) {
  const {
    algorithm = "bm25",
    top = 20,
    profile = "balanced",
    bm25K1 = 1.5,
    bm25B = 0.75,
    tfScheme = "log",
    maxSnippets = 3,
    enabled,
    staleTime = 60_000,
  } = opts;
  return useQuery({
    enabled: (enabled ?? true) && query.trim().length > 0,
    staleTime,
    placeholderData: (prev) => prev,
    queryKey: [
      "lyric",
      query,
      algorithm,
      top,
      profile,
      bm25K1,
      bm25B,
      tfScheme,
      maxSnippets,
    ],
    queryFn: async () => {
      const { data } = await api.get<LyricSearchResponse>("/search/lyric", {
        params: {
          q: query,
          top,
          algorithm,
          profile,
          bm25_k1: bm25K1,
          bm25_b: bm25B,
          tf_scheme: tfScheme,
          max_snippets: maxSnippets,
        },
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

export function useAlbum(id: string | null | undefined, opts: Opts<AlbumResponse> = {}) {
  const { enabled, staleTime = 60_000 } = opts;
  return useQuery({
    enabled: (enabled ?? true) && !!id,
    staleTime,
    queryKey: ["album", id],
    queryFn: async () => {
      const { data } = await api.get<AlbumResponse>(`/album/${id}`);
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
