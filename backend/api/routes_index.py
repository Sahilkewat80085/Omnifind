from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.session import get_db
from models.schemas.file_schemas import IndexStats
from models.schemas.index_schemas import (
    IndexStatusResponse,
    ScanRequest,
    WatchedFolderResponse,
    WatchFolderRequest,
)
from services.folder_watcher_service import FolderWatcherService, get_watcher_service
from services.index_job_manager import IndexJobManager, get_job_manager
from services.metadata_service import MetadataService

router = APIRouter(prefix="/index", tags=["index"])


@router.post("/scan")
def start_scan(
    request: ScanRequest,
    job_manager: IndexJobManager = Depends(get_job_manager),
    watcher_service: FolderWatcherService = Depends(get_watcher_service),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    if not Path(request.path).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid folder: {request.path}")

    # Register as watched folder for real-time automatic background indexing
    metadata_service = MetadataService(db)
    metadata_service.add_watched_folder(request.path)
    watcher_service.watch_folder(request.path)

    started = job_manager.start(request.path)
    if not started:
        raise HTTPException(status_code=409, detail="An indexing job is already running")
    return {"status": "started"}


@router.get("/status", response_model=IndexStatusResponse)
def get_status(job_manager: IndexJobManager = Depends(get_job_manager)) -> IndexStatusResponse:
    state = job_manager.get_state()
    return IndexStatusResponse(
        is_running=state.is_running,
        processed=state.processed,
        total=state.total,
        current_file=state.current_file,
        last_error=state.last_error,
        indexed_count=state.indexed_count,
        removed_count=state.removed_count,
    )


@router.get("/stats", response_model=IndexStats)
def get_stats(db: Session = Depends(get_db)) -> IndexStats:
    return MetadataService(db).get_stats()


# --- Watched Folders Endpoints ---


@router.post("/watch", response_model=WatchedFolderResponse)
def add_watch_folder(
    request: WatchFolderRequest,
    watcher_service: FolderWatcherService = Depends(get_watcher_service),
    job_manager: IndexJobManager = Depends(get_job_manager),
    db: Session = Depends(get_db),
) -> WatchedFolderResponse:
    if not Path(request.path).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid folder: {request.path}")

    metadata_service = MetadataService(db)
    folder_resp = metadata_service.add_watched_folder(request.path)
    watcher_service.watch_folder(request.path)

    # Trigger initial scan if not already running
    job_manager.start(request.path)

    return folder_resp


@router.get("/watched-folders", response_model=list[WatchedFolderResponse])
def get_watched_folders(db: Session = Depends(get_db)) -> list[WatchedFolderResponse]:
    return MetadataService(db).list_watched_folders()


@router.delete("/watch")
def remove_watch_folder(
    path: str,
    watcher_service: FolderWatcherService = Depends(get_watcher_service),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    metadata_service = MetadataService(db)
    watcher_service.unwatch_folder(path)
    removed = metadata_service.remove_watched_folder(path)
    if not removed:
        raise HTTPException(status_code=404, detail="Folder is not currently watched")
    return {"status": "unwatched", "path": path}
