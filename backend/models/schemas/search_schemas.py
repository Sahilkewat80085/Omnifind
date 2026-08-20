from typing import Union
from pydantic import BaseModel, Field

from models.schemas.file_schemas import FileType


class DocumentResult(BaseModel):
    result_type: FileType = FileType.document
    file_id: str
    file_name: str
    path: str
    similarity: float
    page_number: int | None
    chunk_text: str
    chunk_index: int
    match_source: str = "semantic"  # "lexical" | "semantic" | "hybrid"
    duplicate_count: int = 0
    duplicate_files: list[str] = Field(default_factory=list)


class ImageResult(BaseModel):
    result_type: FileType = FileType.image
    file_id: str
    file_name: str
    path: str
    similarity: float
    width: int
    height: int
    match_source: str = "semantic"
    duplicate_count: int = 0
    duplicate_files: list[str] = Field(default_factory=list)


class CodeResult(BaseModel):
    """A source-code hit."""

    result_type: FileType = FileType.code
    file_id: str
    file_name: str
    path: str
    similarity: float
    language: str
    symbol: str | None
    line_start: int
    line_end: int
    chunk_text: str
    chunk_index: int
    match_source: str = "semantic"
    duplicate_count: int = 0
    duplicate_files: list[str] = Field(default_factory=list)


SearchResult = Union[DocumentResult, ImageResult, CodeResult]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    filtered_to: FileType | None = None
