import { useSearch, useLyricSearch, useArtist, useSong } from "../api/hooks";
import type { ArtistResponse, SongResponse } from "../api/types";
import { resolveIntent, type ResolvedIntent } from "../utils/intent";

export interface SearchFlowResult {
  query: string;
  intent: ResolvedIntent;
  isSearching: boolean;
  isEnriching: boolean;
  isError: boolean;
  error: unknown;
  search: ReturnType<typeof useSearch>;
  artist?: ArtistResponse;
  song?: SongResponse;
  lyric: ReturnType<typeof useLyricSearch>;
}

export function useSearchFlow(query: string): SearchFlowResult {
  const search = useSearch(query);
  const intent = resolveIntent(search.data);
  const topItem = search.data?.items?.[0];
  const topId = topItem?.id;

  const artistQuery = useArtist(topId, {
    enabled: intent === "artist" && !!topId,
  });

  const songQuery = useSong(topId, {
    enabled: intent === "track" && !!topId,
  });

  const lyric = useLyricSearch(query, {
    enabled: intent === "lyric" && query.trim().length > 0,
  });

  const isEnriching =
    (intent === "artist" && artistQuery.isLoading) ||
    (intent === "track" && songQuery.isLoading) ||
    (intent === "lyric" && lyric.isLoading);

  return {
    query,
    intent,
    isSearching: search.isLoading,
    isEnriching: isEnriching ?? false,
    isError: search.isError || artistQuery.isError || songQuery.isError || lyric.isError,
    error: search.error ?? artistQuery.error ?? songQuery.error ?? lyric.error,
    search,
    artist: artistQuery.data,
    song: songQuery.data,
    lyric,
  };
}
