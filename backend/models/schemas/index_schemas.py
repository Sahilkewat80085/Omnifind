from datetime import datetime
from pydantic import BaseModel, ConfigDict


class ScanRequest(BaseModel):
    path: str


class IndexStatusResponse(BaseModel):
    is_running: bool
    processed: int
    total: int
    current_file: str | None = None
    last_error: str | None = None
    indexed_count: int = 0
    removed_count: int = 0


class WatchFolderRequest(BaseModel):
    path: str


class WatchedFolderResponse(BaseModel):
    id: str
    path: str
    is_active: bool
    added_at: datetime
    last_scanned_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class WatcherActivityResponse(BaseModel):
    file_name: str
    path: str
    action: str
    timestamp: float
