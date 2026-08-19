import type {
  AskResponse,
  FileMetadata,
  Health,
  IndexStats,
  IndexStatus,
  SearchResponse,
} from "./types";

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";
const STORAGE_KEY = "omnifind.backendUrl";

export class ApiError extends Error {
  // Callers branch on status — notably 503 (no Gemini key) and 409 (indexing
  // already running), which are expected states rather than real failures.
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function getBaseUrl(): string {
  return localStorage.getItem(STORAGE_KEY) ?? DEFAULT_BASE_URL;
}

export function setBaseUrl(url: string): void {
  localStorage.setItem(STORAGE_KEY, url.replace(/\/+$/, ""));
}

export function resetBaseUrl(): void {
  localStorage.removeItem(STORAGE_KEY);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(getBaseUrl() + path, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    // fetch only rejects on network-level failure, which here almost always
    // means the backend isn't running. Say that instead of "Failed to fetch".
    throw new ApiError(
      "Cannot reach the backend. Is uvicorn running on " + getBaseUrl() + "?",
      0,
    );
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON error body; the status text is the best we have.
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),

  stats: () => request<IndexStats>("/index/stats"),

  indexStatus: () => request<IndexStatus>("/index/status"),

  startIndexing: (path: string) =>
    request<{ status: string }>("/index/scan", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  search: (query: string) =>
    request<SearchResponse>("/search?q=" + encodeURIComponent(query)),

  ask: (query: string, topK?: number) =>
    request<AskResponse>("/ask", {
      method: "POST",
      body: JSON.stringify({ q: query, top_k: topK ?? null }),
    }),

  listFiles: () => request<FileMetadata[]>("/files"),

  openFile: (path: string) =>
    request<{ status: string }>("/files/open", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  // Not a fetch: this is a URL for <img src>. The browser can't load local
  // file:// paths from an http page, so image previews stream through the
  // backend instead.
  fileContentUrl: (fileId: string) => `${getBaseUrl()}/files/${fileId}/raw`,
};
