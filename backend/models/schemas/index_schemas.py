from pydantic import BaseModel


class ScanRequest(BaseModel):
    path: str


class IndexStatusResponse(BaseModel):
    is_running: bool
    processed: int
    total: int
    current_file: str | None = None
    last_error: str | None = None
    indexed_count: int = 0
    # Files that were in the index but no longer on disk, cleared by the last
    # scan. Surfaced so a disappearing search result is explained rather than
    # looking like the index lost something on its own.
    removed_count: int = 0
