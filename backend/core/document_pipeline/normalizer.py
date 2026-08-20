import re
from core.document_pipeline.models import NormalizedBlock, NormalizedDocument


class DocumentNormalizer:
    """Cleans, normalizes text blocks, and standardizes structural hierarchies."""

    def normalize(self, doc: NormalizedDocument) -> NormalizedDocument:
        cleaned_blocks: list[NormalizedBlock] = []

        for block in doc.blocks:
            # 1. Normalize Whitespace & linebreaks
            clean_text = self._clean_text(block.text)
            if not clean_text:
                continue

            cleaned_blocks.append(
                NormalizedBlock(
                    text=clean_text,
                    location=block.location,
                    heading_path=block.heading_path,
                    block_type=block.block_type,
                    level=block.level,
                )
            )

        doc.blocks = cleaned_blocks
        return doc

    def _clean_text(self, text: str) -> str:
        # Replace non-breaking spaces and irregular unicode spaces
        text = text.replace("\u00a0", " ").replace("\u200b", "")
        # Normalize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse excessive vertical whitespace (> 2 blank lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple horizontal whitespace
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()
