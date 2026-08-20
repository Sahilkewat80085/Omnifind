from functools import lru_cache
from pathlib import Path
from typing import Any

import open_clip
import torch
from PIL import Image

from core.embeddings.errors import ModelsNotAvailableError
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Predefined zero-shot concept taxonomy for visual understanding
VISUAL_CONCEPTS = [
    # Primary Subject / Category
    ("Portrait & Face", "a portrait photo of a person, headshot, or face"),
    ("Passport & ID Photo", "a passport photo, identification card, or formal headshot"),
    ("Landscape & Nature", "a scenic outdoor landscape, nature, sky, trees, mountains or forest"),
    ("Mountains & Hills", "a photo of mountains, rocky peaks, valleys or hill terrain"),
    ("Beach & Ocean", "a photo of a beach, ocean, sea, lake or tropical water"),
    ("City & Architecture", "a photo of city buildings, urban streets, bridges, or modern architecture"),
    ("Room & Interior", "an indoor photograph of a room, living space, office, or furniture"),
    ("Document & Text Scan", "a document, printed paper, receipt, certificate, diagram, or text"),
    ("Digital Art & Graphic", "digital artwork, illustration, vector graphics, meme, or computer wallpaper"),
    ("Code & UI Screenshot", "a computer screen screenshot showing source code, editor, or app interface"),
    ("Food & Cuisine", "a photograph of delicious food, plated dish, restaurant meal, or beverage"),
    ("Animal & Wildlife", "a photo of an animal, cute pet, dog, cat, bird, or wildlife"),
    ("Vehicle & Transport", "a photo of a car, automobile, motorcycle, airplane, or vehicle"),
    ("Objects & Still Life", "a close-up photograph of an object, gadget, book, or everyday item"),
    # Environment & Lighting
    ("Bright Daylight", "a photo taken in bright sunny daylight with clear natural light"),
    ("Night & Low Light", "a photo taken at night, in darkness, or under moody low lighting"),
    ("Studio & Formal Lighting", "a photo with studio lighting, neutral background, or clean contrast"),
]


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


@lru_cache(maxsize=1)
def _get_cached_concept_embeddings() -> tuple[list[tuple[str, str]], torch.Tensor]:
    model, _, tokenizer = _get_model_bundle()
    prompts = [c[1] for c in VISUAL_CONCEPTS]
    tokens = tokenizer(prompts)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return VISUAL_CONCEPTS, features


def _extract_dominant_colors(img: Image.Image, num_colors: int = 3) -> list[dict[str, Any]]:
    """Extract dominant palette colors with hex code and approximate naming."""
    try:
        small = img.convert("RGB").resize((64, 64), Image.Resampling.NEAREST)
        palette = small.quantize(colors=num_colors, method=Image.Quantize.FASTOCTREE)
        palette_colors = palette.getpalette()
        if not palette_colors:
            return []

        colors: list[dict[str, Any]] = []
        for i in range(num_colors):
            r = palette_colors[i * 3]
            g = palette_colors[i * 3 + 1]
            b = palette_colors[i * 3 + 2]
            hex_code = f"#{r:02x}{g:02x}{b:02x}".upper()
            colors.append({"hex": hex_code, "rgb": [r, g, b]})
        return colors
    except Exception:
        return []


def _describe_aspect_ratio(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "Unknown"
    ratio = width / height
    if 0.95 <= ratio <= 1.05:
        return "1:1 Square"
    elif ratio < 0.95:
        if 0.70 <= ratio <= 0.85:
            return "4:5 Portrait"
        elif 0.50 <= ratio <= 0.65:
            return "9:16 Vertical"
        return "Portrait"
    else:
        if 1.25 <= ratio <= 1.40:
            return "4:3 Standard"
        elif 1.65 <= ratio <= 1.85:
            return "16:9 Widescreen"
        return "Landscape"


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

    def understand_image(self, path: str) -> dict[str, Any]:
        """Performs zero-shot visual understanding on an image using OpenCLIP.

        Returns detected visual concepts, confidence scores, scene attributes,
        dominant colors, and an AI-generated semantic summary of the image content.
        """
        path_obj = Path(path)
        if not path_obj.is_file():
            return {"error": "Image file not found"}

        try:
            with Image.open(path_obj) as img:
                width, height = img.size
                format_name = img.format or path_obj.suffix.lstrip(".").upper()
                mode = img.mode
                dominant_colors = _extract_dominant_colors(img, num_colors=3)

                model, preprocess, _ = _get_model_bundle()
                image_input = preprocess(img.convert("RGB")).unsqueeze(0)
                with torch.no_grad():
                    img_features = model.encode_image(image_input)
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            concepts, concept_features = _get_cached_concept_embeddings()
            with torch.no_grad():
                similarities = (img_features @ concept_features.T).squeeze(0).tolist()

            # Rank concepts by similarity
            scored_concepts = []
            for (label, _prompt), raw_sim in zip(concepts, similarities):
                # Calibrate CLIP cosine score (~0.15 - 0.35) to intuitive 0-100% confidence
                confidence = max(0.0, min(1.0, (raw_sim - 0.16) / 0.18))
                scored_concepts.append({
                    "label": label,
                    "confidence": round(confidence * 100, 1),
                    "raw_similarity": round(raw_sim, 4),
                })

            scored_concepts.sort(key=lambda c: c["raw_similarity"], reverse=True)
            top_detected = [c for c in scored_concepts if c["confidence"] >= 35.0][:5]
            if not top_detected and scored_concepts:
                top_detected = scored_concepts[:3]

            aspect_ratio_label = _describe_aspect_ratio(width, height)

            # Generate natural language summary from top concepts
            top_labels = [c["label"] for c in top_detected[:2]]
            if top_labels:
                summary = f"Identified primarily as {top_labels[0]}"
                if len(top_labels) > 1:
                    summary += f" with {top_labels[1]} elements"
                summary += f" ({aspect_ratio_label}, {width}×{height}px)."
            else:
                summary = f"Image content indexed ({aspect_ratio_label}, {width}×{height}px)."

            return {
                "summary": summary,
                "aspect_ratio": aspect_ratio_label,
                "dimensions": f"{width} × {height} px",
                "format": format_name,
                "color_mode": mode,
                "dominant_colors": dominant_colors,
                "detected_concepts": top_detected,
            }
        except Exception:
            logger.exception("Failed to perform visual understanding for: %s", path)
            return {
                "summary": "Visual embedding generated.",
                "detected_concepts": [],
            }
