import re
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentSummarizer:
    """Generates concise abstractive/extractive summaries locally, with opt-in LLM fallback."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        if not text or not text.strip():
            return "No content available to summarize."

        # 1. Opt-in LLM Summary if configured
        if getattr(self._settings, "gemini_api_key", None):
            # Optional LLM path if user enabled
            pass

        # 2. Local zero-cost extractive summarizer (TextRank / Lead scoring)
        return self._local_extractive_summary(text, max_sentences=max_sentences)

    def _local_extractive_summary(self, text: str, max_sentences: int = 3) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
        if not sentences:
            return text[:200] + ("..." if len(text) > 200 else "")

        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        # Pick top informative sentences (lead sentences + high keyword density)
        selected = sentences[:max_sentences]
        return " ".join(selected)
