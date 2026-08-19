import { useState } from "react";

import { api } from "../api/client";
import type { SearchResponse } from "../api/types";
import { Banner } from "../components/Banner";
import { CodeCard } from "../components/CodeCard";
import { EmptyState } from "../components/EmptyState";
import { FileCard } from "../components/FileCard";
import { LoadingIndicator } from "../components/LoadingIndicator";
import { SearchBar } from "../components/SearchBar";
import { presentableResults } from "../utils/relevance";

const EXAMPLES = [
  "how much money was paid",
  "travel destinations brochure",
  "picture of a dog",
];

// Naming a file type in the query filters to it — "mountain image" searches
// pictures only. Plural, because they label a set of results.
const TYPE_LABELS: Record<string, string> = {
  document: "documents",
  image: "images",
  code: "source files",
};

export function SearchPage() {
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleSearch(query: string) {
    setBusy(true);
    setError(null);
    try {
      setResponse(await api.search(query));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResponse(null);
    } finally {
      setBusy(false);
    }
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
        examples={EXAMPLES}
        busy={busy}
        onSubmit={handleSearch}
      />

      {error && (
        <div style={{ marginTop: 20 }}>
          <Banner kind="error">{error}</Banner>
        </div>
      )}

      {busy && <LoadingIndicator label="Embedding query and searching…" />}

      {!busy && response && (() => {
        // Filtered and de-duplicated before anything is counted, so the header
        // never promises more rows than it renders.
        const results = presentableResults(response.results);
        const typeLabel = TYPE_LABELS[response.filtered_to ?? ""] ?? null;
        const typeWord = response.filtered_to ?? "";

        return (
        <>
          {results.length > 0 && (
            <div className="section-label">
              {results.length} result{results.length === 1 ? "" : "s"} for “
              {response.query}”
              {typeLabel && <span className="filter-chip">{typeLabel} only</span>}
            </div>
          )}

          {results.length === 0 ? (
            // When the query named a type, say so. Otherwise "No item matched"
            // looks like broken search rather than an empty category — the
            // user asked for an image and there simply is no matching image.
            <EmptyState
              icon="⌕"
              title={typeLabel ? `No matching ${typeLabel.slice(0, -1)}` : "No item matched"}
              hint={
                typeLabel
                  ? `Your search asked for ${typeLabel}, so only ${typeLabel} were searched. Nothing in that category is a close enough match — drop the word “${typeWord}” to search everything instead.`
                  : "Nothing in the index is a close enough match. Try describing the content differently, or index the folder that contains it."
              }
            />
          ) : (
            results.map((result) =>
              // Narrowing on result_type gives each card its exact shape —
              // code needs line numbers, images need dimensions.
              result.result_type === "code" ? (
                <CodeCard key={result.file_id} result={result} />
              ) : (
                <FileCard key={result.file_id} result={result} />
              ),
            )
          )}
        </>
        );
      })()}
    </>
  );
}
