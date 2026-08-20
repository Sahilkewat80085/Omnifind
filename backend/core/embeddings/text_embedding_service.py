from functools import lru_cache

from sentence_transformers import SentenceTransformer

from core.embeddings.errors import ModelsNotAvailableError
from utils.config import get_settings
from utils.logger import get_logger
from utils.offline import model_downloads_allowed

logger = get_logger(__name__)

# bge-small-en-v1.5 is an asymmetric retrieval model: passages are encoded
# plain, but queries need this instruction prefix or recall drops sharply.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    settings = get_settings()
    logger.info("Loading text embedding model: %s", settings.text_embedding_model)
    try:
        # Belt and braces alongside HF_HUB_OFFLINE. The env var has to be set
        # before huggingface_hub is imported to have any effect, which is easy
        # to break from a script or a test; this argument is read here and now,
        # so it holds whatever the import order turned out to be.
        return SentenceTransformer(
            settings.text_embedding_model,
            local_files_only=not model_downloads_allowed(),
        )
    except Exception as exc:
        raise ModelsNotAvailableError(settings.text_embedding_model, exc) from exc


class TextEmbeddingService:
    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_model()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def encode_query(self, text: str) -> list[float]:
        model = _get_model()
        vector = model.encode(QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()
