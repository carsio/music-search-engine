import { type FormEvent } from "react";

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit: (v: string) => void;
  compact?: boolean;
  placeholder?: string;
}

export function SearchBox({ value, onChange, onSubmit, compact, placeholder }: Props) {
  const submit = (ev: FormEvent) => {
    ev.preventDefault();
    onSubmit(value);
  };
  return (
    <form className={`search-box ${compact ? "search-box--compact" : ""}`} onSubmit={submit}>
      <input
        type="search"
        className="search-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder ?? "O que você quer ouvir?"}
        autoFocus={!compact}
      />
      <button type="submit" className="search-submit">
        buscar
      </button>
    </form>
  );
}
