import { useState } from "react";

interface Props {
  placeholder: string;
  buttonLabel: string;
  busy?: boolean;
  onSubmit: (query: string) => void;
}

export function SearchBar({
  placeholder,
  buttonLabel,
  busy = false,
  onSubmit,
}: Props) {
  const [value, setValue] = useState("");

  function submit(query: string) {
    const trimmed = query.trim();
    if (trimmed && !busy) onSubmit(trimmed);
  }

  return (
    <form
      className="row"
      onSubmit={(e) => {
        e.preventDefault();
        submit(value);
      }}
    >
      <input
        className="input"
        value={value}
        placeholder={placeholder}
        onChange={(e) => setValue(e.target.value)}
      />
      <button className="btn" type="submit" disabled={busy || !value.trim()}>
        {busy ? "Working…" : buttonLabel}
      </button>
    </form>
  );
}
