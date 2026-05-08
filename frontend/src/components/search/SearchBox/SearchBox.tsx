import { useEffect, useRef, useState, type FormEvent } from "react";
import { Icon } from "../../primitives/Icon";
import { cx } from "../../../utils/classnames";
import styles from "./SearchBox.module.css";

interface SearchBoxProps {
  value: string;
  onChange: (next: string) => void;
  onSubmit: (value: string) => void;
  variant?: "hero" | "header";
  placeholder?: string;
  autoFocus?: boolean;
  ctaLabel?: string;
}

export function SearchBox({
  value,
  onChange,
  onSubmit,
  variant = "header",
  placeholder = "buscar artista, música, letra ou gênero…",
  autoFocus,
  ctaLabel,
}: SearchBoxProps) {
  const [focused, setFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (autoFocus) {
      const id = requestAnimationFrame(() => inputRef.current?.focus());
      return () => cancelAnimationFrame(id);
    }
  }, [autoFocus]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = value.trim();
    if (trimmed) onSubmit(trimmed);
  }

  return (
    <form
      className={cx(styles.box, variant === "hero" && styles.hero, focused && styles.focused)}
      onSubmit={handleSubmit}
      role="search"
    >
      <span className={styles.icon} aria-hidden>
        <Icon name="search" size={variant === "hero" ? 22 : 18} />
      </span>
      <input
        ref={inputRef}
        className={styles.input}
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        placeholder={placeholder}
        aria-label="Buscar"
      />
      {value ? (
        <button
          type="button"
          className={styles.clear}
          onClick={() => onChange("")}
          aria-label="Limpar busca"
        >
          <Icon name="close" size={16} />
        </button>
      ) : null}
      <button
        type="submit"
        className={cx(styles.submit, variant === "hero" && styles.submitLg)}
        aria-label="Buscar"
      >
        {variant === "hero" && ctaLabel ? (
          ctaLabel
        ) : (
          <Icon name="arrow-right" size={18} />
        )}
      </button>
    </form>
  );
}
