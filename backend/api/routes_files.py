import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from models.schemas.file_schemas import OpenFileRequest

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/exists")
def check_file_exists(path: str = Query(...)) -> dict[str, bool | str]:
    file_path = Path(path)
    exists = file_path.is_file()
    return {"path": path, "exists": exists, "message": "File exists" if exists else "File not found"}


@router.post("/open")
def open_file(request: OpenFileRequest) -> dict[str, str]:
    file_path = Path(request.path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if sys.platform == "win32":
        os.startfile(file_path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(file_path)], check=False)
    else:
        subprocess.run(["xdg-open", str(file_path)], check=False)

    return {"status": "opened"}
