import { useCallback, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import type { SearchAlgorithm } from "../api/types";

export type SearchProfile = "balanced" | "lyrics" | "metadata";
export type SearchTfScheme = "raw" | "log" | "augmented";

export interface SearchSettings {
  algorithm: SearchAlgorithm;
  top: number;
  rerank: boolean;
  profile: SearchProfile;
  bm25K1: number;
  bm25B: number;
  tfScheme: SearchTfScheme;
  maxSnippets: number;
}

const STORAGE_KEY = "music-search:search-settings";

const PARAMS = {
  algorithm: "algo",
  top: "top",
  rerank: "rerank",
  profile: "profile",
  bm25K1: "k1",
  bm25B: "b",
  tfScheme: "tf",
  maxSnippets: "snippets",
} as const;

export const DEFAULT_SEARCH_SETTINGS: SearchSettings = {
  algorithm: "bm25",
  top: 10,
  rerank: false,
  profile: "balanced",
  bm25K1: 1.5,
  bm25B: 0.75,
  tfScheme: "log",
  maxSnippets: 3,
};

function isAlgorithm(value: string | null | undefined): value is SearchAlgorithm {
  return value === "bm25" || value === "tfidf";
}

function isProfile(value: string | null | undefined): value is SearchProfile {
  return value === "balanced" || value === "lyrics" || value === "metadata";
}

function isTfScheme(value: string | null | undefined): value is SearchTfScheme {
  return value === "raw" || value === "log" || value === "augmented";
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function parseIntParam(
  value: string | null | undefined,
  fallback: number,
  min: number,
  max: number,
) {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) return fallback;
  return clamp(parsed, min, max);
}

function parseFloatParam(
  value: string | null | undefined,
  fallback: number,
  min: number,
  max: number,
) {
  if (!value) return fallback;
  const parsed = Number.parseFloat(value);
  if (Number.isNaN(parsed)) return fallback;
  return clamp(parsed, min, max);
}

function parseBooleanParam(value: string | null | undefined, fallback: boolean) {
  if (!value) return fallback;
  if (value === "1" || value === "true") return true;
  if (value === "0" || value === "false") return false;
  return fallback;
}

function readStoredSettings(): Partial<SearchSettings> {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw) as Partial<SearchSettings>;
    return {
      algorithm: isAlgorithm(parsed.algorithm) ? parsed.algorithm : undefined,
      top:
        typeof parsed.top === "number"
          ? clamp(Math.round(parsed.top), 1, 50)
          : undefined,
      rerank: typeof parsed.rerank === "boolean" ? parsed.rerank : undefined,
      profile: isProfile(parsed.profile) ? parsed.profile : undefined,
      bm25K1:
        typeof parsed.bm25K1 === "number"
          ? clamp(parsed.bm25K1, 0.1, 3)
          : undefined,
      bm25B:
        typeof parsed.bm25B === "number"
          ? clamp(parsed.bm25B, 0, 1)
          : undefined,
      tfScheme: isTfScheme(parsed.tfScheme) ? parsed.tfScheme : undefined,
      maxSnippets:
        typeof parsed.maxSnippets === "number"
          ? clamp(Math.round(parsed.maxSnippets), 1, 10)
          : undefined,
    };
  } catch {
    return {};
  }
}

function resolveSettings(params: URLSearchParams): SearchSettings {
  const stored = readStoredSettings();
  const algorithmParam = params.get(PARAMS.algorithm);
  const profileParam = params.get(PARAMS.profile);
  const tfSchemeParam = params.get(PARAMS.tfScheme);
  return {
    algorithm: isAlgorithm(algorithmParam)
      ? algorithmParam
      : (stored.algorithm ?? DEFAULT_SEARCH_SETTINGS.algorithm),
    top: parseIntParam(
      params.get(PARAMS.top),
      stored.top ?? DEFAULT_SEARCH_SETTINGS.top,
      1,
      50,
    ),
    rerank: parseBooleanParam(
      params.get(PARAMS.rerank),
      stored.rerank ?? DEFAULT_SEARCH_SETTINGS.rerank,
    ),
    profile: isProfile(profileParam)
      ? profileParam
      : (stored.profile ?? DEFAULT_SEARCH_SETTINGS.profile),
    bm25K1: parseFloatParam(
      params.get(PARAMS.bm25K1),
      stored.bm25K1 ?? DEFAULT_SEARCH_SETTINGS.bm25K1,
      0.1,
      3,
    ),
    bm25B: parseFloatParam(
      params.get(PARAMS.bm25B),
      stored.bm25B ?? DEFAULT_SEARCH_SETTINGS.bm25B,
      0,
      1,
    ),
    tfScheme: isTfScheme(tfSchemeParam)
      ? tfSchemeParam
      : (stored.tfScheme ?? DEFAULT_SEARCH_SETTINGS.tfScheme),
    maxSnippets: parseIntParam(
      params.get(PARAMS.maxSnippets),
      stored.maxSnippets ?? DEFAULT_SEARCH_SETTINGS.maxSnippets,
      1,
      10,
    ),
  };
}

function writeStoredSettings(settings: SearchSettings) {
  if (typeof window === "undefined") {
    return;
  }
  window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

export function useSearchSettings() {
  const [params, setParams] = useSearchParams();
  const settings = resolveSettings(params);
  const hasActiveOverrides =
    settings.algorithm !== DEFAULT_SEARCH_SETTINGS.algorithm ||
    settings.top !== DEFAULT_SEARCH_SETTINGS.top ||
    settings.rerank !== DEFAULT_SEARCH_SETTINGS.rerank ||
    settings.profile !== DEFAULT_SEARCH_SETTINGS.profile ||
    settings.bm25K1 !== DEFAULT_SEARCH_SETTINGS.bm25K1 ||
    settings.bm25B !== DEFAULT_SEARCH_SETTINGS.bm25B ||
    settings.tfScheme !== DEFAULT_SEARCH_SETTINGS.tfScheme ||
    settings.maxSnippets !== DEFAULT_SEARCH_SETTINGS.maxSnippets;

  useEffect(() => {
    writeStoredSettings(settings);
  }, [settings]);

  const updateSettings = useCallback(
    (patch: Partial<SearchSettings>) => {
      const next = { ...settings, ...patch };
      writeStoredSettings(next);
      setParams((prev) => {
        const updated = new URLSearchParams(prev);
        updated.set(PARAMS.algorithm, next.algorithm);
        updated.set(PARAMS.top, String(next.top));
        updated.set(PARAMS.rerank, next.rerank ? "1" : "0");
        updated.set(PARAMS.profile, next.profile);
        updated.set(PARAMS.bm25K1, String(next.bm25K1));
        updated.set(PARAMS.bm25B, String(next.bm25B));
        updated.set(PARAMS.tfScheme, next.tfScheme);
        updated.set(PARAMS.maxSnippets, String(next.maxSnippets));
        return updated;
      });
    },
    [setParams, settings],
  );

  const resetSettings = useCallback(() => {
    if (typeof window !== "undefined") {
      window.sessionStorage.removeItem(STORAGE_KEY);
    }
    setParams((prev) => {
      const updated = new URLSearchParams(prev);
      updated.delete(PARAMS.algorithm);
      updated.delete(PARAMS.top);
      updated.delete(PARAMS.rerank);
      updated.delete(PARAMS.profile);
      updated.delete(PARAMS.bm25K1);
      updated.delete(PARAMS.bm25B);
      updated.delete(PARAMS.tfScheme);
      updated.delete(PARAMS.maxSnippets);
      return updated;
    });
  }, [setParams]);

  return {
    settings,
    hasActiveOverrides,
    updateSettings,
    resetSettings,
  };
}