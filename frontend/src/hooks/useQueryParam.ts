import { useCallback } from "react";
import { useSearchParams } from "react-router-dom";

export function useQueryParam(key: string): [string, (next: string) => void] {
  const [params, setParams] = useSearchParams();
  const value = params.get(key) ?? "";

  const setValue = useCallback(
    (next: string) => {
      setParams((prev) => {
        const updated = new URLSearchParams(prev);
        if (next) {
          updated.set(key, next);
        } else {
          updated.delete(key);
        }
        return updated;
      });
    },
    [key, setParams],
  );

  return [value, setValue];
}
