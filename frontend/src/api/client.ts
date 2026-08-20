import type {
  AskResponse,
  FileIndexDetail,
  FileMetadata,
  Health,
  IndexStats,
  IndexStatus,
  SearchResponse,
  WatchedFolder,
  WatcherActivity,
} from "./types";

/**
 * Where the API lives when the user has not overridden it in Settings.
 *
 * In a production build the backend serves this bundle itself, so the API is
 * on the same origin and the correct answer is "wherever this page came from"
 * — the desktop launcher binds an OS-assigned port, and hard-coding 8000 would
 * point the app at a port nothing is listening on. In dev the bundle is served
 * by Vite on :5173 while the backend runs separately, so the default has to
 * name that backend explicitly.
 */
const DEFAULT_BASE_URL = import.meta.env.DEV
  ? "http://127.0.0.1:8000"
  : window.location.origin;

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

  watcherActivity: (since = 0) =>
    request<WatcherActivity[]>(`/index/activity?since=${since}`),

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

  fileIndexDetails: (fileId: string) =>
    request<FileIndexDetail>(`/files/${fileId}/index-details`),

  openFile: (path: string) =>
    request<{ status: string }>("/files/open", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  fileContentUrl: (fileId: string) => `${getBaseUrl()}/files/${fileId}/raw`,
};
