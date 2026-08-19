from pathlib import Path

from core.parsers.base import PageText


def parse_txt(path: str) -> list[PageText]:
    text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [PageText(page_number=None, text=text)]
