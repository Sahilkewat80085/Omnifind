import { useEffect, useState } from "react";

import { ApiError, api } from "../api/client";
import type { IndexStatus, WatchedFolder } from "../api/types";
import { Banner } from "../components/Banner";

const POLL_INTERVAL_MS = 1200;

export function IndexPage() {
  const [path, setPath] = useState("");
  const [status, setStatus] = useState<IndexStatus | null>(null);
  const [watchedFolders, setWatchedFolders] = useState<WatchedFolder[]>([]);
  const [message, setMessage] = useState<{ kind: "success" | "warning" | "error"; text: string } | null>(
    null,
  );
  const [starting, setStarting] = useState(false);

  function loadWatchedFolders() {
    api
      .listWatchedFolders()
      .then(setWatchedFolders)
      .catch(() => {});
  }

  useEffect(() => {
    let cancelled = false;

    function poll() {
      api
        .indexStatus()
        .then((next) => {
          if (!cancelled) setStatus(next);
        })
        .catch(() => {});
    }

    poll();
    loadWatchedFolders();

    const id = setInterval(() => {
      poll();
      loadWatchedFolders();
    }, POLL_INTERVAL_MS);

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
      setMessage({
        kind: "success",
        text: "Folder indexed and added to live auto-indexing monitoring.",
      });
      loadWatchedFolders();
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

  async function handleRemoveWatch(folderPath: string) {
    try {
      await api.removeWatchFolder(folderPath);
      loadWatchedFolders();
      setMessage({ kind: "success", text: `Stopped monitoring: ${folderPath}` });
    } catch (err) {
      setMessage({ kind: "error", text: err instanceof Error ? err.message : String(err) });
    }
  }

  const percent =
    status && status.total > 0
      ? Math.round((status.processed / status.total) * 100)
      : 0;

  return (
    <>
      <header className="page-header">
        <h1>Automatic Folder Monitoring</h1>
        <p className="subtitle">
          Add any folder to start indexing. Folders are automatically monitored in real-time — any file added, updated, or removed will automatically index without manual triggering.
        </p>
      </header>

      <div className="card">
        <label className="field-label" htmlFor="folder-path">
          Folder to watch & auto-index
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
            {starting ? "Starting…" : "Watch & Index"}
          </button>
        </div>
        <p className="muted" style={{ fontSize: 12.5, marginTop: 8 }}>
          Paste a full path. The backend auto-indexes files as they are saved or moved into this directory.
        </p>
      </div>

      {message && (
        <div style={{ marginTop: 16 }}>
          <Banner kind={message.kind}>{message.text}</Banner>
        </div>
      )}

      <div className="section-label">Active Monitored Folders</div>
      <div className="card">
        {watchedFolders.length === 0 ? (
          <p className="muted">No folders are currently being monitored.</p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {watchedFolders.map((folder) => (
              <div
                key={folder.id}
                className="row"
                style={{
                  justifyContent: "space-between",
                  padding: "8px 12px",
                  background: "var(--color-surface-hover)",
                  borderRadius: 4,
                }}
              >
                <div>
                  <strong className="mono" style={{ fontSize: 13.5 }}>
                    📁 {folder.path}
                  </strong>
                  <span
                    style={{
                      marginLeft: 10,
                      fontSize: 11,
                      color: "var(--color-success, #22c55e)",
                      background: "rgba(34, 197, 94, 0.15)",
                      padding: "2px 6px",
                      borderRadius: 3,
                    }}
                  >
                    ● Auto-indexing active
                  </span>
                </div>
                <button
                  className="btn btn-secondary"
                  style={{ fontSize: 12, padding: "4px 10px" }}
                  onClick={() => handleRemoveWatch(folder.path)}
                >
                  Unwatch
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="section-label">Progress</div>

      <div className="card">
        {!status || (status.total === 0 && !status.is_running) ? (
          <p className="muted">No initial scan job is currently running.</p>
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
            {!status.is_running && status.removed_count > 0 && (
              <p className="muted" style={{ marginTop: 10 }}>
                Removed {status.removed_count}{" "}
                {status.removed_count === 1 ? "entry" : "entries"} for files that are no
                longer on disk.
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
