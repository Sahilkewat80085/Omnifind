"""Whether the embedding models are usable, answered at startup rather than
in the middle of the user's first search.

Without this the only way to find out that setup never completed is to type a
query and get a 500. Loading both models at boot turns that into a line in the
log and a flag on /health, and has a second payoff: the first search no longer
pays the several-second model load, because the caches are already warm.
"""

import threading
from dataclasses import dataclass
from typing import Literal

from utils.logger import get_logger

logger = get_logger(__name__)

# Tri-state, not a boolean. Warm-up takes a few seconds, and during those
# seconds "not ready" is not the same claim as "setup is broken" — collapsing
# the two flashes a scary banner across every single startup.
ModelState = Literal["pending", "ready", "unavailable"]


@dataclass(frozen=True)
class ModelStatus:
    state: ModelState
    detail: str

    @property
    def ready(self) -> bool:
        return self.state == "ready"


_status = ModelStatus(state="pending", detail="Loading embedding models...")
_lock = threading.Lock()


def get_status() -> ModelStatus:
    return _status


def warm_up() -> ModelStatus:
    """Load both models, recording whether they came up.

    Imports are local so that merely importing this module does not drag in
    torch — `utils/offline.py` must get its environment variables set before
    anything touches huggingface_hub, and a top-level import here would run
    too early to guarantee that.

    Encoding one short string rather than just constructing the models is
    deliberate: a model can load and still fail on first use if a tokenizer
    file is missing from a half-finished download.
    """
    global _status

    with _lock:
        from core.embeddings.image_embedding_service import ImageEmbeddingService
        from core.embeddings.text_embedding_service import TextEmbeddingService

        try:
            TextEmbeddingService().encode_query("warm up")
            ImageEmbeddingService().encode_text("warm up")
        except Exception as exc:
            _status = ModelStatus(state="unavailable", detail=str(exc))
            logger.error("Embedding models are not ready: %s", exc)
        else:
            _status = ModelStatus(
                state="ready", detail="Embedding models loaded from the local cache."
            )
            logger.info("Embedding models warmed up - search will run fully offline")

    return _status
