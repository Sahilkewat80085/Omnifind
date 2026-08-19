from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.session import get_db
from models.schemas.file_schemas import IndexStats
from models.schemas.index_schemas import IndexStatusResponse, ScanRequest
from services.index_job_manager import IndexJobManager, get_job_manager
from services.metadata_service import MetadataService

router = APIRouter(prefix="/index", tags=["index"])


@router.post("/scan")
def start_scan(
    request: ScanRequest,
    job_manager: IndexJobManager = Depends(get_job_manager),
) -> dict[str, str]:
    if not Path(request.path).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid folder: {request.path}")

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
