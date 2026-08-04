from pydantic import BaseModel

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
    exists: bool = True


class ImageResult(BaseModel):
    result_type: FileType = FileType.image
    file_id: str
    file_name: str
    path: str
    similarity: float
    width: int
    height: int
    exists: bool = True


class SearchResponse(BaseModel):
    query: str
    results: list[DocumentResult | ImageResult]
