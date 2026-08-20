"""Rebuild the SQLite `chunks` table from what Qdrant already holds.

Strict literal matching reads passage text out of SQLite, but an index built
before that table was written has vectors and no text: 52 files, 0 chunks. The
search then finds nothing but file names and looks broken.

Re-indexing would fix it and costs a full re-embed of every file. Nothing was
actually lost, though - every chunk's text is sitting in its Qdrant payload, so
this copies it back across. Images are skipped: they have no text to store.

    backend/.venv/Scripts/python.exe -m scripts.backfill_chunks
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.vectorstore.qdrant_client import VectorService  # noqa: E402
from database.session import SessionLocal  # noqa: E402
from models.schemas.file_schemas import FileType  # noqa: E402
from services.metadata_service import MetadataService  # noqa: E402


def backfill() -> int:
    """Returns how many files had their chunk rows written."""
    db = SessionLocal()
    try:
        meta = MetadataService(db)
        known = {f.path: f for f in meta.list_files()}

        by_path: dict[str, list[dict]] = defaultdict(list)
        for payload in VectorService().iter_payloads():
            if payload.get("file_type") == FileType.image.value:
                continue
            if not payload.get("chunk_text"):
                continue
            path = str(Path(str(payload.get("path", ""))).resolve())
            by_path[path].append(payload)

        written = 0
        for path, payloads in by_path.items():
            file_meta = known.get(path)
            if file_meta is None:
                print(f"  skip (not in metadata): {path}")
                continue

            # upsert_chunks clears the path's rows first, so running this twice
            # is a no-op rather than a duplicate.
            payloads.sort(key=lambda p: int(p.get("chunk_index", 0)))
            file_type = FileType(payloads[0].get("file_type", FileType.document.value))

            meta.upsert_chunks(
                file_id=file_meta.id,
                file_name=file_meta.file_name,
                file_type=file_type,
                path=path,
                chunks=[
                    {
                        "chunk_index": int(p.get("chunk_index", 0)),
                        "page_number": p.get("page_number"),
                        "line_start": p.get("line_start"),
                        "line_end": p.get("line_end"),
                        "symbol": p.get("symbol"),
                        "language": p.get("language"),
                        "chunk_text": p.get("chunk_text", ""),
                    }
                    for p in payloads
                ],
            )
            written += 1
            print(f"  {len(payloads):4d} chunks  {file_meta.file_name}")

        return written
    finally:
        db.close()


if __name__ == "__main__":
    print("Restoring passage text from the vector store...")
    count = backfill()
    print(f"\nDone. {count} file(s) now have searchable passage text.")
    if count == 0:
        print("Nothing to restore - re-index a folder if search still finds only names.")
