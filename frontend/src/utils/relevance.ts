import type { SearchResult } from "../api/types";

export const MIN_MATCH_SIMILARITY = 0.15;

export function isRelevant(similarity: number): boolean {
  return similarity >= MIN_MATCH_SIMILARITY;
}

/**
 * One row per file, keeping its best-scoring passage.
 *
 * Results arrive sorted by score, so the first time a file appears is already
 * its strongest hit.
 */
export function dedupeByFile(results: SearchResult[]): SearchResult[] {
  const seen = new Set<string>();
  return results.filter((result) => {
    if (seen.has(result.file_id)) return false;
    seen.add(result.file_id);
    return true;
  });
}

/** What the Search page renders: all unique matched files. */
export function presentableResults(results: SearchResult[]): SearchResult[] {
  return dedupeByFile(results);
}
