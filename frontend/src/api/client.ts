import type {
  AskResponse,
  FileMetadata,
  Health,
  IndexStats,
  IndexStatus,
  SearchResponse,
  WatchedFolder,
} from "./types";

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";
const STORAGE_KEY = "omnifind.backendUrl";

export class ApiError extends Error {
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
      // Non-JSON error body
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

  listWatchedFolders: () => request<WatchedFolder[]>("/index/watched-folders"),

  addWatchFolder: (path: string) =>
    request<WatchedFolder>("/index/watch", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  removeWatchFolder: (path: string) =>
    request<{ status: string; path: string }>(
      `/index/watch?path=${encodeURIComponent(path)}`,
      {
        method: "DELETE",
      },
    ),

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

  fileContentUrl: (fileId: string) => `${getBaseUrl()}/files/${fileId}/raw`,
};
