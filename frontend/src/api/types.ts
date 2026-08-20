// Mirrors backend/models/schemas/*.py. Keep the two in sync — these are the
// contract between the FastAPI backend and this UI.

/** "pending" only during the few seconds of start-up warm-up. */
export type ModelState = "pending" | "ready" | "unavailable";

export interface Health {
  status: string;
  app: string;
  env: string;
  ai_enabled: boolean;
  model: string;
  /**
   * Whether the embedding models are loaded. OmniFind runs offline, but the
   * weights are downloaded once at setup — "unavailable" means that step never
   * happened, and neither search nor indexing can work until it does.
   */
  models_state: ModelState;
  models_detail: string;
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

export interface VectorChunkInfo {
  id: string;
  chunk_index: number | null;
  page_number: number | null;
  line_start: number | null;
  line_end: number | null;
  symbol: string | null;
  language: string | null;
  chunk_text: string | null;
  vector_name: string;
  vector_dimensions: number;
  vector_sample: number[];
}

export interface DetectedConcept {
  label: string;
  confidence: number;
  raw_similarity: number;
}

export interface DominantColor {
  hex: string;
  rgb: number[];
}

export interface VisualUnderstanding {
  summary: string;
  aspect_ratio: string;
  dimensions: string;
  format: string;
  color_mode: string;
  dominant_colors: DominantColor[];
  detected_concepts: DetectedConcept[];
}

export interface FileIndexDetail {
  file_id: string;
  file_name: string;
  file_type: FileTypeName;
  path: string;
  size_bytes: number;
  indexed_at: string;
  chunk_count: number;
  image_width: number | null;
  image_height: number | null;
  language: string | null;
  index_model_info: string;
  visual_understanding?: VisualUnderstanding | null;
  chunks: VectorChunkInfo[];
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

export interface WatchedFolder {
  id: string;
  path: string;
  is_active: boolean;
  added_at: string;
  last_scanned_at: string | null;
}

export interface WatcherActivity {
  file_name: string;
  path: string;
  action: "indexed" | "removed";
  timestamp: number;
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

export type SearchResult = DocumentResult | ImageResult | CodeResult;

export interface SearchResponse {
  query: string;
  results: SearchResult[];
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
  language: string | null;
  symbol: string | null;
  line_start: number | null;
  line_end: number | null;
}

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
  used_context: boolean;
}
