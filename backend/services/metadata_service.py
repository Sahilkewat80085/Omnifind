from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import FileRecord, WatchedFolder
from models.schemas.file_schemas import FileMetadata, FileType, IndexStats
from models.schemas.index_schemas import WatchedFolderResponse
from utils.logger import get_logger

logger = get_logger(__name__)


class MetadataService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert_file(
        self,
        *,
        file_name: str,
        file_type: FileType,
        extension: str,
        path: str,
        size_bytes: int,
        chunk_count: int | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        language: str | None = None,
    ) -> FileMetadata:
        record = self.db.scalar(select(FileRecord).where(FileRecord.path == path))
        if record is None:
            record = FileRecord(
                file_name=file_name,
                file_type=file_type.value,
                extension=extension,
                path=path,
                size_bytes=size_bytes,
            )
            self.db.add(record)
            logger.info("Registering new file: %s", path)
        else:
            record.file_name = file_name
            record.file_type = file_type.value
            record.extension = extension
            record.size_bytes = size_bytes
            logger.info("Re-indexing existing file: %s", path)

        record.chunk_count = chunk_count
        record.image_width = image_width
        record.image_height = image_height
        record.language = language

        self.db.commit()
        self.db.refresh(record)
        return FileMetadata.model_validate(record)

    def get_by_path(self, path: str) -> FileMetadata | None:
        record = self.db.scalar(select(FileRecord).where(FileRecord.path == path))
        return FileMetadata.model_validate(record) if record else None

    def get_by_id(self, file_id: str) -> FileMetadata | None:
        record = self.db.get(FileRecord, file_id)
        return FileMetadata.model_validate(record) if record else None

    def list_files(self, file_type: FileType | None = None) -> list[FileMetadata]:
        stmt = select(FileRecord)
        if file_type is not None:
            stmt = stmt.where(FileRecord.file_type == file_type.value)
        records = self.db.scalars(stmt).all()
        return [FileMetadata.model_validate(r) for r in records]

    def list_paths(self) -> list[str]:
        return list(self.db.scalars(select(FileRecord.path)).all())

    def delete_by_path(self, path: str) -> bool:
        record = self.db.scalar(select(FileRecord).where(FileRecord.path == path))
        if record is None:
            return False
        self.db.delete(record)
        self.db.commit()
        return True

    def delete_paths(self, paths: Sequence[str]) -> int:
        if not paths:
            return 0
        records = self.db.scalars(select(FileRecord).where(FileRecord.path.in_(paths))).all()
        for record in records:
            self.db.delete(record)
        self.db.commit()
        return len(records)

    def get_stats(self) -> IndexStats:
        total_files = self.db.scalar(select(func.count(FileRecord.id))) or 0
        total_documents = (
            self.db.scalar(
                select(func.count(FileRecord.id)).where(FileRecord.file_type == FileType.document.value)
            )
            or 0
        )
        total_images = (
            self.db.scalar(
                select(func.count(FileRecord.id)).where(FileRecord.file_type == FileType.image.value)
            )
            or 0
        )
        total_code = (
            self.db.scalar(
                select(func.count(FileRecord.id)).where(FileRecord.file_type == FileType.code.value)
            )
            or 0
        )
        total_chunks = self.db.scalar(select(func.coalesce(func.sum(FileRecord.chunk_count), 0))) or 0
        total_size_bytes = self.db.scalar(select(func.coalesce(func.sum(FileRecord.size_bytes), 0))) or 0

        return IndexStats(
            total_files=total_files,
            total_documents=total_documents,
            total_images=total_images,
            total_code=total_code,
            total_chunks=total_chunks,
            total_size_bytes=total_size_bytes,
        )

    # --- Watched Folder Operations ---

    def add_watched_folder(self, path: str) -> WatchedFolderResponse:
        resolved = str(Path(path).resolve())
        folder = self.db.scalar(select(WatchedFolder).where(WatchedFolder.path == resolved))
        if folder is None:
            folder = WatchedFolder(path=resolved, is_active=True)
            self.db.add(folder)
            logger.info("Added watched folder: %s", resolved)
        else:
            folder.is_active = True
            logger.info("Re-activated watched folder: %s", resolved)
        self.db.commit()
        self.db.refresh(folder)
        return WatchedFolderResponse.model_validate(folder)

    def remove_watched_folder(self, path: str) -> bool:
        resolved = str(Path(path).resolve())
        folder = self.db.scalar(select(WatchedFolder).where(WatchedFolder.path == resolved))
        if folder is None:
            return False
        self.db.delete(folder)
        self.db.commit()
        logger.info("Removed watched folder: %s", resolved)
        return True

    def list_watched_folders(self, only_active: bool = True) -> list[WatchedFolderResponse]:
        stmt = select(WatchedFolder)
        if only_active:
            stmt = stmt.where(WatchedFolder.is_active.is_(True))
        folders = self.db.scalars(stmt).all()
        return [WatchedFolderResponse.model_validate(f) for f in folders]

    def update_watched_folder_scanned(self, path: str) -> None:
        resolved = str(Path(path).resolve())
        folder = self.db.scalar(select(WatchedFolder).where(WatchedFolder.path == resolved))
        if folder is not None:
            folder.last_scanned_at = datetime.now(timezone.utc)
            self.db.commit()
