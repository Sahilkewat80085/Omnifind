from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class NormalizedBlock(BaseModel):
    """A unit of extracted text with rich structural back-references."""
    text: str
    location: str  # e.g., "Page 3", "Slide 4", "Sheet: Financials, Range: A1:D10"
    heading_path: list[str] = Field(default_factory=list)  # e.g., ["Introduction", "Scope"]
    block_type: str = "paragraph"  # "paragraph", "heading", "list_item", "speaker_note", "code"
    level: int | None = None  # Heading level (1-6)


class NormalizedTable(BaseModel):
    """Structured 2D tabular data with location context."""
    table_id: str
    location: str  # e.g., "Page 2", "Slide 7", "Sheet: Summary"
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: str | None = None


class FormatMetadata(BaseModel):
    """Format-specific document metadata."""
    page_count: int | None = None
    slide_count: int | None = None
    sheet_names: list[str] | None = None
    heading_outline: list[dict[str, Any]] | None = None
    has_speaker_notes: bool | None = None


class NormalizedDocument(BaseModel):
    """Common intermediate representation output by all format extractors."""
    file_path: str
    file_name: str
    file_type: str  # "txt", "md", "pdf", "docx", "pptx", "xlsx"
    created_at: datetime
    modified_at: datetime
    blocks: list[NormalizedBlock] = Field(default_factory=list)
    tables: list[NormalizedTable] = Field(default_factory=list)
    format_metadata: FormatMetadata = Field(default_factory=FormatMetadata)
    is_scanned: bool = False
    requires_ocr: bool = False


class EntityItem(BaseModel):
    text: str
    type: str
    start: int
    end: int


class TableContextObject(BaseModel):
    table_id: str
    location: str
    structured_data: list[list[str]]
    table_description: str
    embedding: list[float] = Field(default_factory=list)


class ChunkContextObject(BaseModel):
    chunk_id: str
    text: str
    location: str
    heading_context: str | None = None
    embedding: list[float] = Field(default_factory=list)
    chunk_summary: str | None = None


class DocumentContextObject(BaseModel):
    """Final output context schema ready for Vector DB & Metadata DB ingestion."""
    file_id: str
    file_path: str
    file_name: str
    file_type: str
    created_at: str
    modified_at: str
    language: str
    word_count: int
    is_scanned: bool = False
    format_metadata: dict[str, Any]
    document_embedding: list[float] = Field(default_factory=list)
    document_summary: str
    context_description: str
    keywords: list[str] = Field(default_factory=list)
    entities: list[EntityItem] = Field(default_factory=list)
    tables: list[TableContextObject] = Field(default_factory=list)
    chunks: list[ChunkContextObject] = Field(default_factory=list)
