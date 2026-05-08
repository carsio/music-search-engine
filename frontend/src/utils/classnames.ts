type ClassValue = string | number | false | null | undefined | Record<string, unknown>;

export function cx(...args: ClassValue[]): string {
  const parts: string[] = [];
  for (const arg of args) {
    if (!arg) continue;
    if (typeof arg === "string" || typeof arg === "number") {
      parts.push(String(arg));
    } else if (typeof arg === "object") {
      for (const [key, value] of Object.entries(arg)) {
        if (value) parts.push(key);
      }
    }
  }
  return parts.join(" ");
}
