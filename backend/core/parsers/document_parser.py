from collections.abc import Callable

from core.parsers.base import PageText
from core.parsers.docx_parser import parse_docx
from core.parsers.pdf_parser import parse_pdf
from core.parsers.txt_parser import parse_txt

_PARSERS: dict[str, Callable[[str], list[PageText]]] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_txt,
}


def parse_document(path: str, extension: str) -> list[PageText]:
    parser = _PARSERS.get(extension.lower())
    if parser is None:
        raise ValueError(f"Unsupported document extension: {extension}")
    return parser(path)
