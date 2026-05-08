import type { ReactNode } from "react";

const STOP = new Set(["de", "do", "da", "dos", "das", "e", "o", "a", "os", "as", "que", "em"]);

function tokenize(query: string): string[] {
  return query
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .split(/\s+/)
    .filter((t) => t.length >= 3 && !STOP.has(t));
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function highlight(text: string, query: string): ReactNode {
  if (!text) return null;
  const tokens = tokenize(query);
  if (tokens.length === 0) return text;
  const pattern = new RegExp(`(${tokens.map(escapeRegex).join("|")})`, "gi");
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  const normalized = text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  let m: RegExpExecArray | null;
  while ((m = pattern.exec(normalized)) !== null) {
    const start = m.index;
    const end = start + m[0].length;
    if (start > lastIndex) parts.push(text.slice(lastIndex, start));
    parts.push(<mark key={`${start}-${end}`}>{text.slice(start, end)}</mark>);
    lastIndex = end;
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts;
}

export function highlightHtml(htmlString: string): ReactNode {
  const parts: ReactNode[] = [];
  const regex = /<mark>(.*?)<\/mark>/gi;
  let lastIndex = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = regex.exec(htmlString)) !== null) {
    if (m.index > lastIndex) parts.push(htmlString.slice(lastIndex, m.index));
    parts.push(<mark key={key++}>{m[1]}</mark>);
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < htmlString.length) parts.push(htmlString.slice(lastIndex));
  return parts;
}
