import { useState } from "react";

import { api } from "../api/client";
import type { DocumentResult, ImageResult } from "../api/types";

interface Props {
  result: DocumentResult | ImageResult;
}

export function FileCard({ result }: Props) {
  const [openError, setOpenError] = useState<string | null>(null);
  const [imageFailed, setImageFailed] = useState(false);

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
          <span>{result.result_type === "document" ? "📄" : "🖼"}</span>
          <span className="result-name">{result.file_name}</span>
          {result.result_type === "document" && result.page_number !== null && (
            <span className="result-meta">page {result.page_number}</span>
          )}
          {result.result_type === "image" && (
            <span className="result-meta">
              {result.width} × {result.height}
            </span>
          )}
        </div>

        <div className="result-path">{result.path}</div>

        {result.result_type === "document" ? (
          <p className="result-snippet">{result.chunk_text}</p>
        ) : (
          !imageFailed && (
            <img
              className="thumb"
              src={api.fileContentUrl(result.file_id)}
              alt={result.file_name}
              loading="lazy"
              onError={() => setImageFailed(true)}
            />
          )
        )}

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
