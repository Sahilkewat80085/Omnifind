import { useState } from "react";

import { api } from "../api/client";
import type { FileTypeName, SearchResponse } from "../api/types";
import { Banner } from "../components/Banner";
import { CodeCard } from "../components/CodeCard";
import { EmptyState } from "../components/EmptyState";
import { FileCard } from "../components/FileCard";
import { LoadingIndicator } from "../components/LoadingIndicator";
import { SearchBar } from "../components/SearchBar";
import { presentResults } from "../utils/relevance";

const TYPE_FILTERS: Array<{ label: string; value: FileTypeName | null }> = [
  { label: "All types", value: null },
  { label: "Documents", value: "document" },
  { label: "Images", value: "image" },
  { label: "Code", value: "code" },
];

const TYPE_LABELS: Record<string, string> = {
  document: "Documents",
  image: "Images",
  code: "Code",
};

interface SearchOutcome {
  response: SearchResponse;
  requestedType: FileTypeName | null;
}

const PAGE_SIZE = 10;
const FETCH_LIMIT = 200;

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [fileType, setFileType] = useState<FileTypeName | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<SearchOutcome | null>(null);
  const [page, setPage] = useState(1);

  async function runSearch(text: string, type: FileTypeName | null) {
    setBusy(true);
    setError(null);
    setPage(1);
    try {
      const response = await api.search(text, { limit: FETCH_LIMIT, fileType: type });
      setOutcome({ response, requestedType: type });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setOutcome(null);
    } finally {
      setBusy(false);
    }
  }

  function handleSearch(text: string) {
    setQuery(text);
    void runSearch(text, fileType);
  }

  function handleFilter(type: FileTypeName | null) {
    setFileType(type);
    if (query) void runSearch(query, type);
  }

  return (
    <>
      <header className="page-header">
        <h1>Search</h1>
        <p className="subtitle">
          Ask in plain English. This matches meaning, not filenames — documents and
          images are embedded by different models but ranked together in one list.
          Point it at a repository and it searches source code the same way.
        </p>
      </header>

      <SearchBar
        placeholder="e.g. how much did I pay in fees?"
        buttonLabel="Search"
        busy={busy}
        onSubmit={handleSearch}
      />

      <div className="chip-row" style={{ marginTop: 12 }}>
        <span className="muted" style={{ fontSize: 12.5, alignSelf: "center" }}>
          File type:
        </span>
        {TYPE_FILTERS.map((filter) => (
          <button
            key={filter.label}
            className={"chip" + (fileType === filter.value ? " active" : "")}
            type="button"
            aria-pressed={fileType === filter.value}
            disabled={busy}
            onClick={() => handleFilter(filter.value)}
          >
            {filter.label}
          </button>
        ))}
      </div>

      {error && (
        <div style={{ marginTop: 20 }}>
          <Banner kind="error">{error}</Banner>
        </div>
      )}

      {busy && <LoadingIndicator label="Embedding query and searching…" />}

      {!busy && outcome && (() => {
        const { response, requestedType } = outcome;
        const { ordered } = presentResults(response.results);
        const totalPages = Math.max(1, Math.ceil(ordered.length / PAGE_SIZE));
        const currentPage = Math.min(page, totalPages);
        const startIndex = (currentPage - 1) * PAGE_SIZE;
        const endIndex = Math.min(startIndex + PAGE_SIZE, ordered.length);
        const visible = ordered.slice(startIndex, endIndex);

        const readFromQuery = response.filtered_to && response.filtered_to !== requestedType;
        const typeWord = response.filtered_to ?? "";
        const typeLabel = TYPE_LABELS[response.filtered_to ?? ""] ?? null;

        return (
          <>
            {ordered.length > 0 && (
              <div className="section-label" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  {totalPages > 1
                    ? `Showing ${startIndex + 1}–${endIndex} of ${ordered.length} results for “${response.query}”`
                    : `${ordered.length} result${ordered.length === 1 ? "" : "s"} for “${response.query}”`}
                  {readFromQuery && typeLabel && (
                    <span className="filter-chip">{typeLabel} only</span>
                  )}
                </div>
                {totalPages > 1 && (
                  <span className="pagination-counter-text">
                    Page {currentPage} of {totalPages}
                  </span>
                )}
              </div>
            )}

            {visible.length === 0 ? (
              <EmptyState
                icon="⌕"
                title={typeLabel ? `No matching ${typeLabel.slice(0, -1)}` : "No item matched"}
                hint={
                  requestedType && typeLabel
                    ? `The ${typeLabel} filter is on, so only ${typeLabel} were searched. Choose “All types” to search everything instead.`
                    : typeLabel
                      ? `Your search asked for ${typeLabel}, so only ${typeLabel} were searched. Nothing in that category is a close enough match — drop the word “${typeWord}” to search everything instead.`
                      : "Nothing in the index is a close enough match. Try describing the content differently, or index the folder that contains it."
                }
              />
            ) : (
              visible.map((result) => (
                <div key={result.file_id}>
                  {result.result_type === "code" ? (
                    <CodeCard result={result} />
                  ) : (
                    <FileCard result={result} />
                  )}
                </div>
              ))
            )}

            {totalPages > 1 && (
              <div className="pagination-bar">
                <button
                  className="btn secondary small pagination-nav-btn"
                  disabled={currentPage === 1}
                  onClick={() => {
                    setPage((p) => Math.max(1, p - 1));
                    window.scrollTo({ top: 0, behavior: "smooth" });
                  }}
                >
                  ← Previous
                </button>

                <div className="pagination-pages">
                  {Array.from({ length: totalPages }, (_, i) => i + 1)
                    .filter((p) => p === 1 || p === totalPages || Math.abs(p - currentPage) <= 2)
                    .map((p, idx, arr) => {
                      const prev = arr[idx - 1];
                      return (
                        <span key={p} className="pagination-item-group">
                          {prev && p - prev > 1 && <span className="pagination-ellipsis">…</span>}
                          <button
                            className={`pagination-page-btn ${p === currentPage ? "active" : ""}`}
                            type="button"
                            onClick={() => {
                              setPage(p);
                              window.scrollTo({ top: 0, behavior: "smooth" });
                            }}
                          >
                            {p}
                          </button>
                        </span>
                      );
                    })}
                </div>

                <button
                  className="btn secondary small pagination-nav-btn"
                  disabled={currentPage === totalPages}
                  onClick={() => {
                    setPage((p) => Math.min(totalPages, p + 1));
                    window.scrollTo({ top: 0, behavior: "smooth" });
                  }}
                >
                  Next →
                </button>
              </div>
            )}
          </>
        );
      })()}
    </>
  );
}
