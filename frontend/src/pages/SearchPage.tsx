import { useState } from "react";

import { api } from "../api/client";
import type { FileTypeName, SearchResponse } from "../api/types";
import { Banner } from "../components/Banner";
import { CodeCard } from "../components/CodeCard";
import { EmptyState } from "../components/EmptyState";
import { FileCard } from "../components/FileCard";
import { LoadingIndicator } from "../components/LoadingIndicator";
import { SearchBar } from "../components/SearchBar";
import { presentableResults } from "../utils/relevance";

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
          Every result contains what you typed. Files <em>named</em> for your search
          come first, then files with it inside them — a file that matches only in
          meaning is not shown. Documents, images and source code are searched
          together, and meaning still decides the order within each group.
        </p>
      </header>

      <SearchBar
        placeholder={'e.g. fee receipt, or "Sinhgad College" for an exact phrase'}
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
        // Order comes from the backend and is meaningful: files named for the
        // query, then files containing it. Re-sorting here by score alone would
        // undo that, so the UI only de-duplicates.
        const ordered = presentableResults(response.results);
        const ignored = response.ignored_terms ?? [];
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

            {ignored.length > 0 && ordered.length > 0 && (
              <div className="muted" style={{ fontSize: 12.5, marginBottom: 10 }}>
                No indexed file contains {ignored.map((t) => `“${t}”`).join(", ")}, so{" "}
                {ignored.length === 1 ? "it was" : "they were"} not required of these results.
              </div>
            )}

            {visible.length === 0 ? (
              <EmptyState
                icon="⌕"
                title={`No file contains “${response.query}”`}
                hint={
                  requestedType && typeLabel
                    ? `Only ${typeLabel} were searched, and none of them contain every word of your search. Choose “All types” to search everything, or search fewer words.`
                    : typeLabel
                      ? `Your search asked for ${typeLabel}, so only ${typeLabel} were searched. Drop the word “${typeWord}” to search everything instead.`
                      : "A file has to contain every word you searched for. Try fewer words, check the spelling, or index the folder that holds the file."
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
