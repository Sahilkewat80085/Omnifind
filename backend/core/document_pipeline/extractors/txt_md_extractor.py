import os
import re
from datetime import datetime, timezone
from pathlib import Path

from core.document_pipeline.extractors.base import BaseExtractor
from core.document_pipeline.models import (
    FormatMetadata,
    NormalizedBlock,
    NormalizedDocument,
    NormalizedTable,
)


class TxtMdExtractor(BaseExtractor):
    """Extractor for plain text (.txt) and Markdown (.md) documents."""

    def extract(self, file_path: str) -> NormalizedDocument:
        path = Path(file_path)
        stat = path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        ext = path.suffix.lower().lstrip(".")

        raw_bytes = path.read_bytes()
        # Encoding normalization
        text = self._decode_text(raw_bytes)

        blocks: list[NormalizedBlock] = []
        tables: list[NormalizedTable] = []
        heading_outline: list[dict[str, int | str]] = []

        if ext == "md":
            blocks, tables, heading_outline = self._parse_markdown(text)
        else:
            blocks = self._parse_plaintext(text)

        return NormalizedDocument(
            file_path=str(path.resolve()),
            file_name=path.name,
            file_type=ext,
            created_at=created_at,
            modified_at=modified_at,
            blocks=blocks,
            tables=tables,
            format_metadata=FormatMetadata(heading_outline=heading_outline or None),
            is_scanned=False,
            requires_ocr=False,
        )

    def _decode_text(self, raw: bytes) -> str:
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    def _parse_plaintext(self, text: str) -> list[NormalizedBlock]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        blocks: list[NormalizedBlock] = []
        for idx, para in enumerate(paragraphs, start=1):
            blocks.append(
                NormalizedBlock(
                    text=para,
                    location=f"Paragraph {idx}",
                    heading_path=[],
                    block_type="paragraph",
                )
            )
        return blocks

    def _parse_markdown(
        self, text: str
    ) -> tuple[list[NormalizedBlock], list[NormalizedTable], list[dict[str, int | str]]]:
        lines = text.splitlines()
        blocks: list[NormalizedBlock] = []
        tables: list[NormalizedTable] = []
        outline: list[dict[str, int | str]] = []

        current_heading_path: list[str] = []
        current_para_lines: list[str] = []
        in_code_block = False
        table_lines: list[str] = []

        def flush_para():
            if current_para_lines:
                para_text = "\n".join(current_para_lines).strip()
                if para_text:
                    blocks.append(
                        NormalizedBlock(
                            text=para_text,
                            location=f"Line {len(blocks) + 1}",
                            heading_path=list(current_heading_path),
                            block_type="paragraph",
                        )
                    )
                current_para_lines.clear()

        def flush_table(table_idx: int):
            if len(table_lines) >= 2:
                headers = [c.strip() for c in table_lines[0].strip("|").split("|")]
                rows = []
                for row_line in table_lines[2:] if "---" in table_lines[1] else table_lines[1:]:
                    cols = [c.strip() for c in row_line.strip("|").split("|")]
                    if cols:
                        rows.append(cols)
                tables.append(
                    NormalizedTable(
                        table_id=f"table-{table_idx}",
                        location=f"Markdown Section: {' > '.join(current_heading_path) or 'Top'}",
                        headers=headers,
                        rows=rows,
                    )
                )
            table_lines.clear()

        table_count = 0
        for line in lines:
            trimmed = line.strip()

            # Code fence toggle
            if trimmed.startswith("```"):
                flush_para()
                in_code_block = not in_code_block
                continue

            if in_code_block:
                current_para_lines.append(line)
                continue

            # Markdown table rows
            if trimmed.startswith("|") and trimmed.endswith("|"):
                flush_para()
                table_lines.append(trimmed)
                continue
            elif table_lines:
                table_count += 1
                flush_table(table_count)

            # Markdown Headings
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", trimmed)
            if heading_match:
                flush_para()
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Adjust heading path
                if level <= len(current_heading_path):
                    current_heading_path = current_heading_path[: level - 1]
                current_heading_path.append(title)
                outline.append({"level": level, "title": title})

                blocks.append(
                    NormalizedBlock(
                        text=title,
                        location=f"Heading Level {level}",
                        heading_path=list(current_heading_path[:-1]),
                        block_type="heading",
                        level=level,
                    )
                )
                continue

            if not trimmed:
                flush_para()
            else:
                current_para_lines.append(line)

        flush_para()
        if table_lines:
            table_count += 1
            flush_table(table_count)

        return blocks, tables, outline
