import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { FileIndexDetail } from "../api/types";
import { formatBytes, formatDate } from "../utils/format";

interface Props {
  fileId: string;
  onClose: () => void;
}

export function IndexInspectorModal({ fileId, onClose }: Props) {
  const [detail, setDetail] = useState<FileIndexDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openStatus, setOpenStatus] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .fileIndexDetails(fileId)
      .then((data) => {
        if (!cancelled) setDetail(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [fileId]);

  async function handleOpenFile(path: string) {
    try {
      await api.openFile(path);
      setOpenStatus("Opened");
      setTimeout(() => setOpenStatus(null), 2500);
    } catch (err) {
      setOpenStatus(err instanceof Error ? err.message : "Failed to open file");
    }
  }

  const fileTypeIcon =
    detail?.file_type === "document" ? "📄" : detail?.file_type === "image" ? "🖼" : "</>";

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <span className="modal-icon">{fileTypeIcon}</span>
            <div>
              <h2 className="modal-title">{detail?.file_name || "Inspecting Vector Index"}</h2>
              <div className="modal-subtitle">{detail?.path}</div>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">
            ✕
          </button>
        </div>

        <div className="modal-body">
          {loading && (
            <div className="modal-loading">
              <span className="spinner" /> Loading visual and vector indexing data from Qdrant…
            </div>
          )}

          {error && <div className="modal-error-banner">⚠️ {error}</div>}

          {!loading && detail && (
            <>
              {/* Summary Stats Grid */}
              <div className="inspector-summary-grid">
                <div className="inspector-stat">
                  <span className="inspector-stat-label">File Type</span>
                  <span className="inspector-stat-value capitalize">{detail.file_type}</span>
                </div>
                <div className="inspector-stat">
                  <span className="inspector-stat-label">File Size</span>
                  <span className="inspector-stat-value">{formatBytes(detail.size_bytes)}</span>
                </div>
                <div className="inspector-stat">
                  <span className="inspector-stat-label">Indexed At</span>
                  <span className="inspector-stat-value">{formatDate(detail.indexed_at)}</span>
                </div>
                <div className="inspector-stat">
                  <span className="inspector-stat-label">Total Chunks</span>
                  <span className="inspector-stat-value">
                    {detail.chunks.length} point{detail.chunks.length === 1 ? "" : "s"}
                  </span>
                </div>
              </div>

              {/* Embedding Model Specs */}
              <div className="inspector-model-badge">
                <span className="model-badge-icon">🧠</span>
                <div>
                  <div className="model-badge-title">Vector Embedding Model</div>
                  <div className="model-badge-desc">{detail.index_model_info}</div>
                </div>
              </div>

              {/* If Image: Show rich visual understanding & concepts detected */}
              {detail.file_type === "image" && (
                <div className="inspector-image-section">
                  <div className="section-label" style={{ marginTop: 16 }}>
                    Visual Understanding & AI Concept Detection
                  </div>

                  <div className="inspector-image-container">
                    <div className="image-preview-wrapper">
                      <img
                        src={api.fileContentUrl(detail.file_id)}
                        alt={detail.file_name}
                        className="inspector-image-preview"
                      />
                      {detail.visual_understanding?.dominant_colors && (
                        <div className="color-palette-bar">
                          {detail.visual_understanding.dominant_colors.map((c) => (
                            <span
                              key={c.hex}
                              className="color-swatch"
                              style={{ backgroundColor: c.hex }}
                              title={`Dominant Color: ${c.hex} RGB(${c.rgb.join(", ")})`}
                            />
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="inspector-image-meta">
                      {detail.visual_understanding && (
                        <div className="understanding-summary-box">
                          <div className="summary-title">👁️ Vision System Comprehension:</div>
                          <div className="summary-text">{detail.visual_understanding.summary}</div>
                        </div>
                      )}

                      {detail.visual_understanding?.detected_concepts &&
                        detail.visual_understanding.detected_concepts.length > 0 && (
                          <div className="concepts-list-section">
                            <div className="concepts-title">Detected Visual Concepts & Confidence:</div>
                            <div className="concepts-grid">
                              {detail.visual_understanding.detected_concepts.map((concept) => (
                                <div key={concept.label} className="concept-item">
                                  <div className="concept-label-row">
                                    <span className="concept-name">{concept.label}</span>
                                    <span className="concept-pct">{concept.confidence}%</span>
                                  </div>
                                  <div className="concept-meter">
                                    <div
                                      className="concept-meter-fill"
                                      style={{ width: `${Math.min(100, concept.confidence)}%` }}
                                    />
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                      {detail.chunks[0] && (
                        <div className="vector-sample-box" style={{ marginTop: 14 }}>
                          <div className="vector-sample-label">
                            Qdrant Point ID · {detail.chunks[0].vector_name} ({detail.chunks[0].vector_dimensions} dims):
                          </div>
                          <code>[{detail.chunks[0].vector_sample.join(", ")}…]</code>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* If Document or Code: Show text chunks & extracted tokens */}
              {detail.file_type !== "image" && (
                <div className="inspector-chunks-section">
                  <div className="section-label" style={{ marginTop: 20 }}>
                    Indexed Vector Chunks ({detail.chunks.length})
                  </div>
                  <p className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
                    These extracted and tokenized excerpts were embedded into 384-dimensional dense vectors
                    and stored in Qdrant for semantic search.
                  </p>

                  <div className="inspector-chunks-list">
                    {detail.chunks.map((chunk, idx) => (
                      <div key={chunk.id || idx} className="inspector-chunk-card">
                        <div className="chunk-card-header">
                          <div className="chunk-badge">
                            Chunk #{chunk.chunk_index !== null ? chunk.chunk_index + 1 : idx + 1}
                          </div>
                          {chunk.page_number !== null && (
                            <span className="result-meta">Page {chunk.page_number}</span>
                          )}
                          {chunk.line_start !== null && (
                            <span className="result-meta">
                              Lines {chunk.line_start}–{chunk.line_end}
                            </span>
                          )}
                          {chunk.symbol && <span className="lang-tag">{chunk.symbol}</span>}
                          {chunk.language && <span className="lang-tag">{chunk.language}</span>}
                          <span className="result-meta" style={{ marginLeft: "auto" }}>
                            {chunk.vector_dimensions} dims
                          </span>
                        </div>

                        <div className="chunk-text-box">
                          {chunk.chunk_text ? (
                            <pre className="chunk-text-content">
                              <code>{chunk.chunk_text}</code>
                            </pre>
                          ) : (
                            <div className="muted">(No text extracted)</div>
                          )}
                        </div>

                        <div className="vector-sample-box">
                          <span className="vector-sample-label">Embedding Vector (Sample):</span>
                          <code>[{chunk.vector_sample.join(", ")}…]</code>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <div className="modal-footer">
          {openStatus && <span className="modal-status-text">{openStatus}</span>}
          {detail && (
            <button className="btn secondary small" onClick={() => handleOpenFile(detail.path)}>
              Open local file
            </button>
          )}
          <button className="btn small" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
