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

export interface PresentedResults {
  /** Every file worth showing, best first: confident matches, then weak ones. */
  ordered: SearchResult[];
  /** How many leading entries of `ordered` clear the presentation floor. */
  confidentCount: number;
}

/**
 * Split what came back into what to show and what to hold behind "explore more".
 */
export function presentResults(results: SearchResult[]): PresentedResults {
  const files = dedupeByFile(results);
  const confident = files.filter((r) => isRelevant(r.similarity));
  const weak = files.filter((r) => !isRelevant(r.similarity));
  return { ordered: [...confident, ...weak], confidentCount: confident.length };
}

/** What the Search page renders: all unique matched files. */
export function presentableResults(results: SearchResult[]): SearchResult[] {
  return dedupeByFile(results);
}
