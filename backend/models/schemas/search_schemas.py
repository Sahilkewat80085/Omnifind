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


class ImageResult(BaseModel):
    result_type: FileType = FileType.image
    file_id: str
    file_name: str
    path: str
    similarity: float
    width: int
    height: int


class CodeResult(BaseModel):
    """A source-code hit.

    Carries line numbers rather than a page number, and the symbol the chunk
    defines, so the UI can show "rag_service.py · lines 106-135 · def
    _retrieve_images" instead of an anonymous slice of a file.
    """

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


class SearchResponse(BaseModel):
    query: str
    results: list[DocumentResult | ImageResult | CodeResult]
    # The file type read out of the query itself, e.g. "mountain image" → image.
    # Surfaced rather than applied silently: a filter the user cannot see is
    # indistinguishable from missing results when the guess is wrong.
    filtered_to: FileType | None = None
