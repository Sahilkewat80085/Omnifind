from datetime import datetime
from enum import Enum
from typing import Any

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


class VectorChunkInfo(BaseModel):
    id: str
    chunk_index: int | None = None
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    symbol: str | None = None
    language: str | None = None
    chunk_text: str | None = None
    vector_name: str
    vector_dimensions: int
    vector_sample: list[float] = []


class DetectedConcept(BaseModel):
    label: str
    confidence: float
    raw_similarity: float


class DominantColor(BaseModel):
    hex: str
    rgb: list[int]


class VisualUnderstanding(BaseModel):
    summary: str
    aspect_ratio: str
    dimensions: str
    format: str
    color_mode: str
    dominant_colors: list[DominantColor] = []
    detected_concepts: list[DetectedConcept] = []


class FileIndexDetail(BaseModel):
    file_id: str
    file_name: str
    file_type: FileType
    path: str
    size_bytes: int
    indexed_at: datetime
    chunk_count: int
    image_width: int | None = None
    image_height: int | None = None
    language: str | None = None
    index_model_info: str
    visual_understanding: VisualUnderstanding | None = None
    chunks: list[VectorChunkInfo] = []


class OpenFileRequest(BaseModel):
    path: str


class IndexStats(BaseModel):
    total_files: int
    total_documents: int
    total_images: int
    total_code: int
    total_chunks: int
    total_size_bytes: int
