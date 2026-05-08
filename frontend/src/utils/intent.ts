import type { SearchResponse } from "../api/types";

export type ResolvedIntent = "artist" | "track" | "lyric" | "genre" | "album" | "none";

export function resolveIntent(response: SearchResponse | undefined): ResolvedIntent {
  if (!response) return "none";
  const used = response.intent_used;
  if (used === "song") return "track";
  if (
    used === "artist" ||
    used === "track" ||
    used === "lyric" ||
    used === "genre" ||
    used === "album"
  ) {
    return used;
  }
  const top = response.items?.[0]?.intent;
  if (top === "artist" || top === "track" || top === "lyric" || top === "genre" || top === "album") {
    return top;
  }
  if (response.items && response.items.length > 0) return "track";
  return "none";
}
