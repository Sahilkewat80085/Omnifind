from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    format: str


def read_image_info(path: str) -> ImageInfo:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return ImageInfo(width=img.width, height=img.height, format=img.format or "")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Not a readable image: {path}") from exc


def extract_text(path: str) -> str | None:
    """OCR hook for a future milestone. Not implemented in this MVP."""
    return None
