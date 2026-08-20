import os
from pathlib import Path
from typing import Type

from core.document_pipeline.extractors.base import BaseExtractor
from core.document_pipeline.extractors.docx_extractor import DocxExtractor
from core.document_pipeline.extractors.pdf_extractor import PdfExtractor
from core.document_pipeline.extractors.pptx_extractor import PptxExtractor
from core.document_pipeline.extractors.txt_md_extractor import TxtMdExtractor
from core.document_pipeline.extractors.xlsx_extractor import XlsxExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


class FormatRouter:
    """Detects document formats via extension & header sniffing and routes to the matching extractor."""

    _EXT_MAP: dict[str, Type[BaseExtractor]] = {
        "txt": TxtMdExtractor,
        "md": TxtMdExtractor,
        "pdf": PdfExtractor,
        "docx": DocxExtractor,
        "pptx": PptxExtractor,
        "xlsx": XlsxExtractor,
    }

    # Magic signatures for binary content sniffing
    _MAGIC_SIGNATURES = {
        b"%PDF-": PdfExtractor,
        b"PK\x03\x04": None,  # ZIP container (DOCX, PPTX, XLSX)
    }

    def get_extractor(self, file_path: str) -> BaseExtractor | None:
        path = Path(file_path)
        if not path.is_file():
            logger.error("File does not exist: %s", file_path)
            return None

        ext = path.suffix.lower().lstrip(".")
        extractor_cls = self._EXT_MAP.get(ext)

        # Content Sniff Verification
        try:
            with open(path, "rb") as f:
                header = f.read(8)

            if header.startswith(b"%PDF-"):
                extractor_cls = PdfExtractor
            elif header.startswith(b"PK\x03\x04"):
                # ZIP container - verify against valid Office XML types
                if ext not in ("docx", "pptx", "xlsx"):
                    extractor_cls = None
        except Exception as e:
            logger.warning("Could not read magic bytes from %s: %s", file_path, e)

        if extractor_cls is None:
            logger.info("Unsupported or unrecognized file format: %s", file_path)
            return None

        return extractor_cls()
