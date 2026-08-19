import type { SearchResult } from "../api/types";

/**
 * Below this, a result is noise dressed up as a percentage.
 *
 * The backend already drops sub-floor image hits, but text hits are only
 * clamped — documents keep their recall on purpose, so a weak match still
 * comes back rather than vanishing. That is the right call for retrieval and
 * the wrong one for a screen: a 12% row reads as an answer simply by being
 * listed. This is the presentation floor, not the retrieval one.
 */
export const MIN_MATCH_SIMILARITY = 0.3;

export function isRelevant(similarity: number): boolean {
  return similarity >= MIN_MATCH_SIMILARITY;
}

/**
 * One row per file, keeping its best-scoring passage.
 *
 * A long PDF or source file produces many chunks, and several can match the
 * same query — which filled the list with the same filename repeated, pushing
 * genuinely different files off the screen. Results arrive sorted by score, so
 * the first time a file appears is already its strongest hit.
 */
export function dedupeByFile(results: SearchResult[]): SearchResult[] {
  const seen = new Set<string>();
  return results.filter((result) => {
    if (seen.has(result.file_id)) return false;
    seen.add(result.file_id);
    return true;
  });
}

/** What the Search page actually renders: relevant, and one row per file. */
export function presentableResults(results: SearchResult[]): SearchResult[] {
  return dedupeByFile(results.filter((r) => isRelevant(r.similarity)));
}
