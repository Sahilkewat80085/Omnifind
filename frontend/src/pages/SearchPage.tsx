import { useState } from "react";

import { api } from "../api/client";
import type { FileTypeName, SearchResponse } from "../api/types";
import { Banner } from "../components/Banner";
import { CodeCard } from "../components/CodeCard";
import { EmptyState } from "../components/EmptyState";
import { FileCard } from "../components/FileCard";
import { LoadingIndicator } from "../components/LoadingIndicator";
import { SearchBar } from "../components/SearchBar";
import { RESULTS_PAGE_SIZE, presentResults } from "../utils/relevance";

// Naming a file type in the query filters to it — "mountain image" searches
// pictures only. Plural, because they label a set of results.
const TYPE_LABELS: Record<string, string> = {
  document: "documents",
  image: "images",
  code: "source files",
};

// The type filter. `null` is "no filter", not a fourth type — the parameter is
// left off the request entirely when nothing is chosen.
const TYPE_FILTERS: { value: FileTypeName | null; label: string }[] = [
  { value: null, label: "All types" },
  { value: "document", label: "Documents" },
  { value: "image", label: "Images" },
  { value: "code", label: "Code" },
];

// Fetched per search, well above the page size: "explore more" pages through
// what is already here rather than firing another query, since re-embedding
// the same question just to reveal row 11 would be slow for no gain.
const FETCH_LIMIT = 50;

interface Outcome {
  response: SearchResponse;
  /** The filter in force when this ran — the live one may already have moved on. */
  requestedType: FileTypeName | null;
}

export function SearchPage() {
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState("");
  const [fileType, setFileType] = useState<FileTypeName | null>(null);
  const [shown, setShown] = useState(RESULTS_PAGE_SIZE);

  // Both arguments are passed in rather than read from state: a filter click
  // sets state and searches in the same handler, where the new value is not
  // visible on `fileType` yet.
  async function runSearch(text: string, type: FileTypeName | null) {
    setBusy(true);
    setError(null);
    // Back to the first page — how far the user had paged through the previous
    // result set means nothing against a different one.
    setShown(RESULTS_PAGE_SIZE);
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
    // Changing the filter with results on screen re-runs the same question at
    // once. Leaving the old list sitting under a new filter would show rows
    // that the filter excludes.
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
        // De-duplicated and split before anything is counted, so the header
        // never promises more rows than it renders.
        const { ordered, confidentCount } = presentResults(response.results);
        const visible = ordered.slice(0, shown);
        const remaining = ordered.length - visible.length;

        // The chip exists to explain a filter the user cannot otherwise see —
        // one read out of their wording. When they picked it from the row
        // above, it is already on screen and repeating it is just noise.
        const readFromQuery = response.filtered_to && response.filtered_to !== requestedType;
        const typeWord = response.filtered_to ?? "";
        const typeLabel = TYPE_LABELS[response.filtered_to ?? ""] ?? null;

        return (
          <>
            {visible.length > 0 && (
              <div className="section-label">
                {remaining > 0
                  ? `${visible.length} of ${ordered.length} results for “${response.query}”`
                  : `${ordered.length} result${ordered.length === 1 ? "" : "s"} for “${response.query}”`}
                {readFromQuery && typeLabel && (
                  <span className="filter-chip">{typeLabel} only</span>
                )}
              </div>
            )}

            {visible.length === 0 ? (
              // When something narrowed the search, say what. Otherwise "No
              // item matched" looks like broken search rather than an empty
              // category — the user asked for an image and there simply is no
              // matching image.
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
              visible.map((result, position) => (
                <div key={result.file_id}>
                  {/* Everything from here down scored below the presentation
                      floor. Labelling the boundary is the point of showing
                      them at all — an unmarked weak match reads as a
                      confident one. */}
                  {position === confidentCount && (
                    <div className="section-label">Weaker matches</div>
                  )}
                  {/* Narrowing on result_type gives each card its exact shape —
                      code needs line numbers, images need dimensions. */}
                  {result.result_type === "code" ? (
                    <CodeCard result={result} />
                  ) : (
                    <FileCard result={result} />
                  )}
                </div>
              ))
            )}

            {remaining > 0 && (
              <div className="more-results">
                <span className="muted">Not seeing the file you had in mind?</span>
                <button
                  className="btn secondary small"
                  type="button"
                  onClick={() => setShown((count) => count + RESULTS_PAGE_SIZE)}
                >
                  Explore {Math.min(remaining, RESULTS_PAGE_SIZE)} more
                </button>
              </div>
            )}
          </>
        );
      })()}
    </>
  );
}
