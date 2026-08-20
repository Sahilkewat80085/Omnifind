import type { SearchResult } from "../api/types";

export const MIN_MATCH_SIMILARITY = 0.15;

/** How many files go on screen at once, and how many each "explore more" adds. */
export const RESULTS_PAGE_SIZE = 10;

export function isRelevant(similarity: number): boolean {
  return similarity >= MIN_MATCH_SIMILARITY;
}

/**
 * One row per file, keeping its best-scoring passage.
 *
 * The backend collapses per file before it trims to top-k - it has to, or a
 * long file's chunks eat every slot and the page is handed a single result.
 * This is the safety net for anything that reaches the UI with duplicates
 * anyway; results arrive sorted, so a file's first appearance is its best hit.
 */
export function dedupeByFile(results: SearchResult[]): SearchResult[] {
  const seen = new Set<string>();
  return results.filter((result) => {
    if (seen.has(result.file_id)) return false;
    seen.add(result.file_id);
    return true;
  });
}

/**
 * What the Search page renders: all unique matched files, in the order given.
 *
 * The rank the backend sends is a tiered one - files named for the query, then
 * files containing it - so the page must not re-sort by score. The old
 * confident/weak split did exactly that and would now shuffle a name match
 * below a passage match. `isRelevant` survives for the Ask page, where a
 * single similarity floor is still the right question to ask.
 */
export function presentableResults(results: SearchResult[]): SearchResult[] {
  return dedupeByFile(results);
}
