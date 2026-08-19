import { useState } from "react";

import { api } from "../api/client";
import type { Citation } from "../api/types";

interface Props {
  citation: Citation;
}

export function ReferenceCard({ citation }: Props) {
  const [openError, setOpenError] = useState<string | null>(null);

  async function handleOpen() {
    setOpenError(null);
    try {
      await api.openFile(citation.path);
    } catch (err) {
      setOpenError(err instanceof Error ? err.message : "Could not open file");
    }
  }

  return (
    <div className="reference-card">
      <div className="reference-marker">{citation.marker}</div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="result-title">
          <span className="result-name">{citation.file_name}</span>
          {citation.line_start !== null ? (
            // A code excerpt is located by line, not page. Saying "page 106"
            // for a function would point at nothing.
            <span className="result-meta">
              lines {citation.line_start}–{citation.line_end}
            </span>
          ) : (
            citation.page_number !== null && (
              <span className="result-meta">page {citation.page_number}</span>
            )
          )}
          {citation.language && <span className="lang-tag">{citation.language}</span>}
          <span className="result-meta">
            {Math.round(citation.similarity * 100)}% match
          </span>
        </div>

        <div className="result-path">{citation.path}</div>
        {citation.line_start !== null ? (
          <pre className="code-snippet reference-code">
            <code>{citation.chunk_text}</code>
          </pre>
        ) : (
          <p className="reference-quote">“{citation.chunk_text}”</p>
        )}

        {openError && (
          <p className="result-meta" style={{ color: "var(--danger)", marginTop: 6 }}>
            {openError}
          </p>
        )}
      </div>

      <button className="btn secondary small" onClick={handleOpen} style={{ alignSelf: "start" }}>
        Open
      </button>
    </div>
  );
}
