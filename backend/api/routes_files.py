import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database.session import get_db
from models.schemas.file_schemas import FileMetadata, FileType, OpenFileRequest
from services.metadata_service import MetadataService

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/exists")
def check_file_exists(path: str = Query(...)) -> dict[str, bool | str]:
    file_path = Path(path)
    exists = file_path.is_file()
    return {"path": path, "exists": exists, "message": "File exists" if exists else "File not found"}


@router.get("", response_model=list[FileMetadata])
def list_files(
    file_type: FileType | None = None,
    db: Session = Depends(get_db),
) -> list[FileMetadata]:
    return MetadataService(db).list_files(file_type)


@router.get("/{file_id}/raw")
def get_file_content(file_id: str, db: Session = Depends(get_db)) -> FileResponse:
    """Serve an indexed file's bytes so the UI can show image thumbnails.

    Deliberately keyed on file_id rather than a path parameter: a browser
    page cannot load local file:// URLs, but an endpoint that served any
    path the caller asked for would read anything on the disk. Resolving
    through the database means only files the user chose to index are
    reachable.
    """
    record = MetadataService(db).get_by_id(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No indexed file with id {file_id}")

    file_path = Path(record.path)
    if not file_path.is_file():
        raise HTTPException(status_code=410, detail=f"File has moved or been deleted: {record.path}")

    return FileResponse(file_path, filename=record.file_name)


@router.post("/open")
def open_file(request: OpenFileRequest) -> dict[str, str]:
    file_path = Path(request.path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {request.path}")

    if sys.platform == "win32":
        os.startfile(file_path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(file_path)], check=False)
    else:
        subprocess.run(["xdg-open", str(file_path)], check=False)

    return {"status": "opened"}
