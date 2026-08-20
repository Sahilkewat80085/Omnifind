from functools import lru_cache

import open_clip
import torch
from PIL import Image

from core.embeddings.errors import ModelsNotAvailableError
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_model_bundle():
    settings = get_settings()
    logger.info(
        "Loading image embedding model: %s (%s)",
        settings.image_embedding_model,
        settings.image_embedding_pretrained,
    )
    # open_clip has no local_files_only switch — it goes through
    # huggingface_hub directly — so HF_HUB_OFFLINE set before import is the
    # only lever here. See utils/offline.py.
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.image_embedding_model, pretrained=settings.image_embedding_pretrained
        )
        tokenizer = open_clip.get_tokenizer(settings.image_embedding_model)
    except Exception as exc:
        name = f"{settings.image_embedding_model} ({settings.image_embedding_pretrained})"
        raise ModelsNotAvailableError(name, exc) from exc
    model.eval()
    return model, preprocess, tokenizer


class ImageEmbeddingService:
    def encode_image(self, path: str) -> list[float]:
        model, preprocess, _ = _get_model_bundle()
        with Image.open(path) as img:
            image_input = preprocess(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            features = model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).tolist()

    def encode_text(self, text: str) -> list[float]:
        model, _, tokenizer = _get_model_bundle()
        tokens = tokenizer([text])
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).tolist()
