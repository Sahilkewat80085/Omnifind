import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from core.vectorstore.qdrant_client import VectorService
from database.session import get_db
from models.schemas.file_schemas import (
    FileIndexDetail,
    FileMetadata,
    FileType,
    OpenFileRequest,
    VectorChunkInfo,
)
from services.metadata_service import MetadataService

router = APIRouter(prefix="/files", tags=["files"])


@router.get("", response_model=list[FileMetadata])
def list_files(
    file_type: FileType | None = None,
    db: Session = Depends(get_db),
) -> list[FileMetadata]:
    return MetadataService(db).list_files(file_type)


@router.get("/{file_id}/index-details", response_model=FileIndexDetail)
def get_file_index_details(file_id: str, db: Session = Depends(get_db)) -> FileIndexDetail:
    """Retrieve detailed vector indexing metadata and chunk breakdown for a file."""
    record = MetadataService(db).get_by_id(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No indexed file with id {file_id}")

    vector_service = VectorService()
    points = vector_service.get_points_by_path(record.path)

    chunks: list[VectorChunkInfo] = []
    for p in points:
        payload = p.get("payload", {})
        vectors = p.get("vectors", {})
        v_name = "text_vector" if record.file_type != FileType.image else "image_vector"
        v_info = vectors.get(v_name) or next(iter(vectors.values()), {"dimensions": 0, "sample": []})

        chunks.append(
            VectorChunkInfo(
                id=p.get("id", ""),
                chunk_index=payload.get("chunk_index"),
                page_number=payload.get("page_number"),
                line_start=payload.get("line_start"),
                line_end=payload.get("line_end"),
                symbol=payload.get("symbol"),
                language=payload.get("language") or record.language,
                chunk_text=payload.get("chunk_text"),
                vector_name=v_name,
                vector_dimensions=v_info.get("dimensions", 0),
                vector_sample=v_info.get("sample", []),
            )
        )

    chunks.sort(key=lambda c: (c.chunk_index if c.chunk_index is not None else 0))

    model_info = (
        "OpenCLIP ViT-B-32 (512-dim visual embeddings · Cosine Distance)"
        if record.file_type == FileType.image
        else "BAAI/bge-small-en-v1.5 (384-dim dense text embeddings · Cosine Distance)"
    )

    return FileIndexDetail(
        file_id=record.id,
        file_name=record.file_name,
        file_type=record.file_type,
        path=record.path,
        size_bytes=record.size_bytes,
        indexed_at=record.indexed_at,
        chunk_count=len(chunks) or (record.chunk_count or 0),
        image_width=record.image_width,
        image_height=record.image_height,
        language=record.language,
        index_model_info=model_info,
        chunks=chunks,
    )


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
