import fitz  # PyMuPDF
from datetime import datetime, timezone
from pathlib import Path

from core.document_pipeline.extractors.base import BaseExtractor
from core.document_pipeline.models import (
    FormatMetadata,
    NormalizedBlock,
    NormalizedDocument,
    NormalizedTable,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class PdfExtractor(BaseExtractor):
    """Extractor for text-based PDFs with layout parsing, tables, and scanned-page detection."""

    def extract(self, file_path: str) -> NormalizedDocument:
        path = Path(file_path)
        stat = path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        blocks: list[NormalizedBlock] = []
        tables: list[NormalizedTable] = []
        doc = fitz.open(str(path))
        page_count = len(doc)

        total_extracted_chars = 0
        table_counter = 0

        for page_num in range(page_count):
            page = doc[page_num]
            location_label = f"Page {page_num + 1}"

            # PyMuPDF text & block extraction
            text_page = page.get_text("blocks")
            page_char_count = 0

            for b in text_page:
                # b = (x0, y0, x1, y1, text, block_no, block_type)
                if len(b) >= 5 and isinstance(b[4], str):
                    block_text = b[4].strip()
                    if block_text:
                        page_char_count += len(block_text)
                        # Determine if heading by single line & length
                        is_short_title = len(block_text.splitlines()) == 1 and len(block_text) < 80
                        blocks.append(
                            NormalizedBlock(
                                text=block_text,
                                location=location_label,
                                heading_path=[],
                                block_type="heading" if is_short_title else "paragraph",
                            )
                        )

            total_extracted_chars += page_char_count

            # Table extraction using PyMuPDF native find_tables
            try:
                page_tables = page.find_tables()
                if page_tables and page_tables.tables:
                    for t in page_tables.tables:
                        table_counter += 1
                        extracted_df = t.extract()
                        if extracted_df and len(extracted_df) > 1:
                            headers = [str(col).strip() if col is not None else "" for col in extracted_df[0]]
                            rows = [
                                [str(cell).strip() if cell is not None else "" for cell in row]
                                for row in extracted_df[1:]
                            ]
                            tables.append(
                                NormalizedTable(
                                    table_id=f"table-{table_counter}",
                                    location=location_label,
                                    headers=headers,
                                    rows=rows,
                                )
                            )
            except Exception as e:
                logger.debug("Table detection skipped on page %d of %s: %s", page_num + 1, path.name, e)

        doc.close()

        # Scanned PDF Detection
        # If total extracted text is negligible (< 10 chars per page on average), flag as scanned
        avg_chars_per_page = total_extracted_chars / max(1, page_count)
        is_scanned = (total_extracted_chars < 20 or avg_chars_per_page < 10)

        if is_scanned:
            logger.warning(
                "PDF %s appears to be scanned/image-only (total chars: %d across %d pages). Flagging requires_ocr.",
                path.name,
                total_extracted_chars,
                page_count,
            )

        return NormalizedDocument(
            file_path=str(path.resolve()),
            file_name=path.name,
            file_type="pdf",
            created_at=created_at,
            modified_at=modified_at,
            blocks=blocks,
            tables=tables,
            format_metadata=FormatMetadata(page_count=page_count),
            is_scanned=is_scanned,
            requires_ocr=is_scanned,
        )
