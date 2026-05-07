// Espelha src/music_search/web/schemas.py.

export type Intent = "artist" | "album" | "song" | "lyric" | "genre" | "none" | "track";
export type SearchAlgorithm = "bm25" | "tfidf";

export interface SearchResultItem {
  id: string;
  rank: number;
  score: number;
  intent: string;
  title: string;
  subtitle?: string | null;
  snippet?: string | null;
  payload: Record<string, unknown>;
}

export interface SearchResponse {
  query: string;
  intent_requested: Intent;
  intent_used: Intent;
  algorithm: SearchAlgorithm;
  items: SearchResultItem[];
  rerank_used: boolean;
  elapsed_ms: number;
}

export interface LyricSnippet {
  line: number;
  text: string;
}

export interface LyricMatch {
  song_id: string;
  title: string;
  artist: string;
  score: number;
  snippets: LyricSnippet[];
}

export interface LyricSearchResponse {
  query: string;
  algorithm: SearchAlgorithm;
  matches: LyricMatch[];
  elapsed_ms: number;
}

export interface AlbumRef {
  title: string;
  year?: number | null;
  tracks?: number | null;
}

export interface TrackRef {
  title: string;
  album?: string | null;
  plays?: string | null;
}

export interface ArtistResponse {
  id: string;
  name: string;
  tagline?: string | null;
  bio?: string | null;
  genres: string[];
  origin?: string | null;
  year_started?: number | null;
  monthly_listeners?: string | null;
  popularity?: number | null;
  albums: AlbumRef[];
  top_tracks: TrackRef[];
  source?: string | null;
  source_url?: string | null;
}

export interface SongResponse {
  id: string;
  title: string;
  artist: string;
  artist_id?: string | null;
  album?: string | null;
  year?: number | null;
  duration?: string | null;
  plays?: string | null;
  composers: string[];
  lyrics?: string | null;
  lyrics_source?: string | null;
  lyrics_source_url?: string | null;
  genres: string[];
  macro_genre?: string | null;
}
