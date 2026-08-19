import { useState } from "react";

import { api } from "../api/client";
import type { CodeResult } from "../api/types";

interface Props {
  result: CodeResult;
}

/**
 * A source-code hit.
 *
 * Rendered in a monospace block that preserves whitespace, because in Python
 * and YAML the indentation *is* the syntax — collapsing it the way the prose
 * snippet does would show something that is not the code in the file.
 */
export function CodeCard({ result }: Props) {
  const [openError, setOpenError] = useState<string | null>(null);

  const percent = Math.round(Math.max(0, Math.min(1, result.similarity)) * 100);
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
            lines {result.line_start}–{result.line_end}
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

      <div className="result-side">
        <div className="match-value">{percent}%</div>
        <div className="match-label">match</div>
        <div className="meter">
          <div className="meter-fill" style={{ width: `${percent}%` }} />
        </div>
        <div className="result-meta" style={{ marginTop: 6 }}>
          {lineCount} line{lineCount === 1 ? "" : "s"}
        </div>
        <button className="btn secondary small" onClick={handleOpen}>
          Open file
        </button>
      </div>
    </div>
  );
}
