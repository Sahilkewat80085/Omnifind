import { useEffect, useState } from "react";

import { ApiError, api } from "../api/client";
import type { IndexStatus } from "../api/types";
import { Banner } from "../components/Banner";

const POLL_INTERVAL_MS = 1200;

export function IndexPage() {
  const [path, setPath] = useState("");
  const [status, setStatus] = useState<IndexStatus | null>(null);
  const [message, setMessage] = useState<{ kind: "success" | "warning" | "error"; text: string } | null>(
    null,
  );
  const [starting, setStarting] = useState(false);

  // Polled continuously while this page is mounted, so progress keeps updating
  // even if the job was kicked off from somewhere else.
  useEffect(() => {
    let cancelled = false;

    function poll() {
      api
        .indexStatus()
        .then((next) => {
          if (!cancelled) setStatus(next);
        })
        .catch(() => {
          /* Sidebar already reports backend reachability; stay quiet here. */
        });
    }

    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  async function handleStart() {
    const trimmed = path.trim();
    if (!trimmed) return;

    setStarting(true);
    setMessage(null);
    try {
      await api.startIndexing(trimmed);
      setMessage({ kind: "success", text: "Indexing started." });
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setMessage({ kind: "warning", text: "An indexing job is already running — see progress below." });
      } else {
        setMessage({ kind: "error", text: err instanceof Error ? err.message : String(err) });
      }
    } finally {
      setStarting(false);
    }
  }

  const percent =
    status && status.total > 0
      ? Math.round((status.processed / status.total) * 100)
      : 0;

  return (
    <>
      <header className="page-header">
        <h1>Index a folder</h1>
        <p className="subtitle">
          Recursively scans a folder for PDF, DOCX, TXT, PNG, JPG and JPEG, then builds
          the semantic index. Re-running on the same folder updates it rather than
          duplicating entries.
        </p>
      </header>

      <div className="card">
        <label className="field-label" htmlFor="folder-path">
          Folder path
        </label>
        <div className="row">
          <input
            id="folder-path"
            className="input"
            value={path}
            placeholder="C:\Users\you\Documents"
            onChange={(e) => setPath(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleStart();
            }}
          />
          <button
            className="btn"
            onClick={handleStart}
            disabled={starting || !path.trim() || status?.is_running}
          >
            {starting ? "Starting…" : "Start indexing"}
          </button>
        </div>
        <p className="muted" style={{ fontSize: 12.5, marginTop: 8 }}>
          Paste a full path. The backend reads the folder directly, so this works for
          any drive the machine can see.
        </p>
      </div>

      {message && (
        <div style={{ marginTop: 16 }}>
          <Banner kind={message.kind}>{message.text}</Banner>
        </div>
      )}

      <div className="section-label">Progress</div>

      <div className="card">
        {!status || (status.total === 0 && !status.is_running) ? (
          <p className="muted">No indexing job has run yet in this session.</p>
        ) : (
          <>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
              <strong>
                {status.is_running ? "Indexing…" : "Completed"} {status.processed} / {status.total} files
              </strong>
              <span className="muted">{percent}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${percent}%` }} />
            </div>
            {status.current_file && status.is_running && (
              <p className="mono" style={{ marginTop: 10 }}>
                {status.current_file}
              </p>
            )}
          </>
        )}

        {status?.last_error && (
          <div style={{ marginTop: 14 }}>
            <Banner kind="error">Last error: {status.last_error}</Banner>
          </div>
        )}
      </div>
    </>
  );
}
