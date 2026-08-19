from functools import lru_cache
from typing import Any

from core.llm.base import LLMError, LLMNotConfiguredError
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _build_client(api_key: str) -> Any:
    """Create the Gemini client once and reuse it.

    The import lives in here, not at module scope, for two reasons: the SDK
    is slow to import (it would tax every backend start, including
    search-only ones), and a machine that hasn't run `pip install` yet
    should still get a working search API instead of a crash at startup.
    """
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise LLMNotConfiguredError(
            "The google-genai package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return genai.Client(api_key=api_key)


class GeminiService:
    """LLMProvider backed by the Google Gemini API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.gemini_api_key.strip()
        self._model = settings.gemini_model
        self._temperature = settings.llm_temperature
        self._max_output_tokens = settings.llm_max_output_tokens

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, *, system_instruction: str, prompt: str) -> str:
        if not self.is_configured:
            raise LLMNotConfiguredError(
                "GEMINI_API_KEY is not set. Add it to backend/.env to enable AI answers."
            )

        from google.genai import types

        client = _build_client(self._api_key)

        try:
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self._temperature,
                    max_output_tokens=self._max_output_tokens,
                ),
            )
        except Exception as exc:
            logger.exception("Gemini generation failed")
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            # Usually a safety block or a max_output_tokens cutoff mid-thought.
            # Either way the caller gets an explicit failure, never a blank answer
            # that looks like the documents had nothing to say.
            raise LLMError("Gemini returned an empty response.")
        return text
