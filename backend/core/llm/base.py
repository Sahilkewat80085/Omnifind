from typing import Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Any failure while generating an answer."""


class LLMNotConfiguredError(LLMError):
    """No API key set, or the provider SDK isn't installed.

    Kept separate from LLMError so the API layer can answer 503 ("this
    install isn't set up for answers yet") instead of 502 ("the provider
    broke"). The two mean very different things to whoever is debugging.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """What RagService needs from a language model, and nothing more.

    RagService depends on this rather than on Gemini directly, so swapping
    providers — or injecting a deterministic fake in tests — touches no
    retrieval or prompting code.
    """

    @property
    def is_configured(self) -> bool:
        """True when a real call could actually be made right now."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier of the model answering, surfaced in API responses."""
        ...

    def generate(self, *, system_instruction: str, prompt: str) -> str:
        """Return the model's reply, or raise an LLMError subclass."""
        ...
