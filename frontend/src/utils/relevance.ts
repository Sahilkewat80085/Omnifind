import type { SearchResult } from "../api/types";

/**
 * Below this, a result is noise dressed up as a percentage.
 *
 * The backend already drops sub-floor image and code hits, but document hits
 * are only clamped - prose keeps its recall on purpose, so a weak match still
 * comes back rather than vanishing. That is the right call for retrieval and
 * the wrong one for the top of a screen: a 12% row reads as an answer simply
 * by being listed. This is the presentation floor, not the retrieval one.
 *
 * It is a fold, not a delete. Weak matches sort below the confident ones and
 * stay collapsed until the user says the file they wanted is not on screen -
 * at which point the recall the backend kept is exactly what they need.
 */
export const MIN_MATCH_SIMILARITY = 0.3;

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
 *
 * Both halves are returned rather than one filtered list, because the page has
 * to label the boundary. An unmarked weak match looks like a confident one,
 * and that is the whole reason the floor exists.
 */
export function presentResults(results: SearchResult[]): PresentedResults {
  const files = dedupeByFile(results);
  const confident = files.filter((r) => isRelevant(r.similarity));
  const weak = files.filter((r) => !isRelevant(r.similarity));
  return { ordered: [...confident, ...weak], confidentCount: confident.length };
}
