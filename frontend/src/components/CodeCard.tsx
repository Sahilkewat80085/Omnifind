import { useState } from "react";

import { api } from "../api/client";
import type { CodeResult } from "../api/types";

interface Props {
  result: CodeResult;
}

export function CodeCard({ result }: Props) {
  const [openError, setOpenError] = useState<string | null>(null);
  const lineCount = result.line_end - result.line_start + 1;

  async function handleOpen() {
    setOpenError(null);
    try {
      await api.openFile(result.path);
    } catch (err) {
      setOpenError(err instanceof Error ? err.message : "Could not open file");
    }
  }

  return (
    <div className="result-card">
      <div className="result-body">
        <div className="result-title">
          <span>{"</>"}</span>
          <span className="result-name">{result.file_name}</span>
          <span className="lang-tag">{result.language}</span>
          <span className="result-meta">
            lines {result.line_start}–{result.line_end} ({lineCount} line{lineCount === 1 ? "" : "s"})
          </span>
        </div>

        <div className="result-path">{result.path}</div>

        {result.symbol && <div className="symbol-line">{result.symbol}</div>}

        <pre className="code-snippet">
          <code>{result.chunk_text}</code>
        </pre>

        {openError && (
          <p className="result-meta" style={{ color: "var(--danger)", marginTop: 8 }}>
            {openError}
          </p>
        )}
      </div>

      <div className="result-side" style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "flex-end" }}>
        <button className="btn secondary small" onClick={handleOpen}>
          Open file
        </button>
      </div>
    </div>
  );
}
