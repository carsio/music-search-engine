import { useSearch, useLyricSearch, useArtist, useAlbum, useSong } from "../api/hooks";
import type { AlbumResponse, ArtistResponse, SongResponse } from "../api/types";
import { resolveIntent, type ResolvedIntent } from "../utils/intent";

export interface SearchFlowResult {
  query: string;
  intent: ResolvedIntent;
  isSearching: boolean;
  isEnriching: boolean;
  isError: boolean;
  error: unknown;
  search: ReturnType<typeof useSearch>;
  album?: AlbumResponse;
  artist?: ArtistResponse;
  song?: SongResponse;
  lyric: ReturnType<typeof useLyricSearch>;
}

export function useSearchFlow(query: string): SearchFlowResult {
  const search = useSearch(query);
  const intent = resolveIntent(search.data);
  const topItem = search.data?.items?.[0];
  const topId = topItem?.id;

  const albumQuery = useAlbum(topId, {
    enabled: intent === "album" && !!topId,
  });

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
    (intent === "album" && albumQuery.isLoading) ||
    (intent === "artist" && artistQuery.isLoading) ||
    (intent === "track" && songQuery.isLoading) ||
    (intent === "lyric" && lyric.isLoading);

  return {
    query,
    intent,
    isSearching: search.isLoading,
    isEnriching: isEnriching ?? false,
    isError: search.isError || albumQuery.isError || artistQuery.isError || songQuery.isError || lyric.isError,
    error: search.error ?? albumQuery.error ?? artistQuery.error ?? songQuery.error ?? lyric.error,
    search,
    album: albumQuery.data,
    artist: artistQuery.data,
    song: songQuery.data,
    lyric,
  };
}
