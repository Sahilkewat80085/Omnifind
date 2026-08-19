from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class FileType(str, Enum):
    document = "document"
    image = "image"
    code = "code"


class FileMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    file_name: str
    file_type: FileType
    extension: str
    path: str
    size_bytes: int
    indexed_at: datetime
    chunk_count: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    language: str | None = None


class OpenFileRequest(BaseModel):
    path: str


class IndexStats(BaseModel):
    total_files: int
    total_documents: int
    total_images: int
    total_code: int
    total_chunks: int
    total_size_bytes: int
