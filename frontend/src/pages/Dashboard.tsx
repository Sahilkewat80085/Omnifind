import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { FileMetadata, IndexStats } from "../api/types";
import { Banner } from "../components/Banner";
import { EmptyState } from "../components/EmptyState";
import { LoadingIndicator } from "../components/LoadingIndicator";
import { StatCard } from "../components/StatCard";
import { formatBytes, formatDate } from "../utils/format";

interface Props {
  onNavigate: (page: "index" | "search") => void;
}

export function Dashboard({ onNavigate }: Props) {
  const [stats, setStats] = useState<IndexStats | null>(null);
  const [files, setFiles] = useState<FileMetadata[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    Promise.all([api.stats(), api.listFiles()])
      .then(([nextStats, nextFiles]) => {
        if (cancelled) return;
        setStats(nextStats);
        setFiles(nextFiles);
        setError(null);
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
  }, []);

  const recent = [...files]
    .sort((a, b) => b.indexed_at.localeCompare(a.indexed_at))
    .slice(0, 8);

  return (
    <>
      <header className="page-header">
        <h1>Dashboard</h1>
        <p className="subtitle">
          Everything below is read live from the local index — nothing here is mocked.
        </p>
      </header>

      {error && <Banner kind="error">{error}</Banner>}
      {loading && <LoadingIndicator label="Reading index…" />}

      {stats && (
        <>
          <div className="stat-grid">
            <StatCard label="Files indexed" value={stats.total_files} />
            <StatCard label="Documents" value={stats.total_documents} />
            <StatCard label="Images" value={stats.total_images} />
            <StatCard label="Source files" value={stats.total_code} />
            <StatCard label="Text chunks" value={stats.total_chunks} />
            <StatCard label="Storage indexed" value={formatBytes(stats.total_size_bytes)} />
          </div>

          {stats.total_files === 0 ? (
            <EmptyState
              icon="⊕"
              title="Nothing indexed yet"
              hint="Go to “Index folder”, point it at a folder of PDFs, DOCX, TXT or images, and OmniFind will build the searchable index."
            />
          ) : (
            <>
              <div className="section-label">Recently indexed</div>
              {recent.map((file) => (
                <div className="reference-card" key={file.id}>
                  <div className="reference-marker">
                    {file.file_type === "document" ? "📄" : "🖼"}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="result-title">
                      <span className="result-name">{file.file_name}</span>
                      <span className="result-meta">{formatBytes(file.size_bytes)}</span>
                      {file.chunk_count !== null && (
                        <span className="result-meta">{file.chunk_count} chunks</span>
                      )}
                      {file.image_width !== null && file.image_height !== null && (
                        <span className="result-meta">
                          {file.image_width} × {file.image_height}
                        </span>
                      )}
                    </div>
                    <div className="result-path">{file.path}</div>
                  </div>
                  <div className="result-meta" style={{ whiteSpace: "nowrap" }}>
                    {formatDate(file.indexed_at)}
                  </div>
                </div>
              ))}

              <div className="divider" />
              <div className="row">
                <button className="btn" onClick={() => onNavigate("search")}>
                  Search the index
                </button>
                <button className="btn secondary" onClick={() => onNavigate("index")}>
                  Index another folder
                </button>
              </div>
            </>
          )}
        </>
      )}
    </>
  );
}
