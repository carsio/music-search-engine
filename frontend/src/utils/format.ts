export function formatNumber(value: number | string | null | undefined): string {
  if (value == null) return "—";
  if (typeof value === "string") return value;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1).replace(".0", "")}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1).replace(".0", "")}k`;
  return String(value);
}

export function formatDuration(seconds: number | null | undefined): string {
  if (!seconds) return "—";
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

export function formatYear(year: number | null | undefined): string {
  if (!year) return "—";
  return String(year);
}

export function placeholder(value: string | number | null | undefined, fallback = "—"): string {
  if (value == null || value === "") return fallback;
  return String(value);
}
