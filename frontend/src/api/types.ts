// Mirrors backend/models/schemas/*.py. Keep the two in sync — these are the
// contract between the FastAPI backend and this UI.

export interface Health {
  status: string;
  app: string;
  env: string;
  ai_enabled: boolean;
  model: string;
}

export type FileTypeName = "document" | "image" | "code";

export interface FileMetadata {
  id: string;
  file_name: string;
  file_type: FileTypeName;
  extension: string;
  path: string;
  size_bytes: number;
  indexed_at: string;
  chunk_count: number | null;
  image_width: number | null;
  image_height: number | null;
  language: string | null;
}

export interface IndexStats {
  total_files: number;
  total_documents: number;
  total_images: number;
  total_code: number;
  total_chunks: number;
  total_size_bytes: number;
}

export interface IndexStatus {
  is_running: boolean;
  processed: number;
  total: number;
  current_file: string | null;
  last_error: string | null;
  indexed_count: number;
  /** Entries cleared because the file is no longer on disk. */
  removed_count: number;
}

export interface DocumentResult {
  result_type: "document";
  file_id: string;
  file_name: string;
  path: string;
  similarity: number;
  page_number: number | null;
  chunk_text: string;
  chunk_index: number;
}

export interface ImageResult {
  result_type: "image";
  file_id: string;
  file_name: string;
  path: string;
  similarity: number;
  width: number;
  height: number;
}

// A source-code hit. Located by line rather than page, and carrying the
// symbol it defines so a result reads as "rag_service.py · lines 106-135 ·
// def _retrieve_images" instead of an anonymous slice of a file.
export interface CodeResult {
  result_type: "code";
  file_id: string;
  file_name: string;
  path: string;
  similarity: number;
  language: string;
  symbol: string | null;
  line_start: number;
  line_end: number;
  chunk_text: string;
  chunk_index: number;
}

// Discriminated on result_type, so narrowing on that field gives the
// component the right shape without any casts.
export type SearchResult = DocumentResult | ImageResult | CodeResult;

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  // The file type read out of the query itself — "mountain image" → "image".
  // null when the query named no type, so everything was searched.
  filtered_to: FileTypeName | null;
}

export interface Citation {
  marker: number;
  file_id: string;
  file_name: string;
  path: string;
  page_number: number | null;
  chunk_text: string;
  similarity: number;
  // Set for code excerpts, which have lines where a document has a page.
  language: string | null;
  symbol: string | null;
  line_start: number | null;
  line_end: number | null;
}

// Not a Citation: images are matched visually by CLIP and never go into the
// prompt, so they have no [n] marker for the answer to reference. The UI keeps
// them in their own section so they don't read as evidence for the answer.
export interface RelatedImage {
  file_id: string;
  file_name: string;
  path: string;
  width: number;
  height: number;
  similarity: number;
}

export interface AskResponse {
  query: string;
  answer: string;
  citations: Citation[];
  related_images: RelatedImage[];
  model: string;
  // Tracks text context only — related_images can be non-empty while this is
  // false, when a question matches pictures but no documents.
  used_context: boolean;
}
