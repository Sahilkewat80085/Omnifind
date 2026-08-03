import docx

from core.parsers.base import PageText


def parse_docx(path: str) -> list[PageText]:
    document = docx.Document(path)
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    if not text:
        return []
    return [PageText(page_number=None, text=text)]
