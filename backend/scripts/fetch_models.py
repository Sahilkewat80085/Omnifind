"""One-time model download. The only part of OmniFind that needs the internet.

Run this once during setup, connected:

    python scripts/fetch_models.py

It pulls the text and image embedding weights into the local HuggingFace cache
(~740 MB). Everything afterwards — indexing, search, the watcher — runs with
the network off; the app forces offline model loading at startup so it can
never quietly depend on the connection again.

Safe to re-run: already-cached files are not downloaded twice, so this doubles
as a way to verify an install.
"""

import os
import sys
from pathlib import Path

# Must be set before anything imports huggingface_hub, which is why it happens
# here at the top of the script rather than in main(): the flag is read into a
# module constant on import, and setting it later does nothing.
os.environ["OMNIFIND_ALLOW_MODEL_DOWNLOAD"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Running `python scripts/fetch_models.py` puts scripts/ on sys.path, not the
# backend root, so the package imports below would fail without this.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))


def _cache_size_mb() -> float:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if not cache.exists():
        return 0.0
    return sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    from utils.config import get_settings

    settings = get_settings()
    print("OmniFind - one-time model download")
    print("This needs an internet connection. The app itself does not.\n")

    before = _cache_size_mb()

    # Imported here, after the environment is set, and encoded once each:
    # downloading the weights is not proof they work, and a half-written
    # tokenizer file only shows up when something actually uses it.
    try:
        print(f"  text model : {settings.text_embedding_model}")
        from core.embeddings.text_embedding_service import TextEmbeddingService

        dim = len(TextEmbeddingService().encode_query("hello"))
        print(f"               ready ({dim}-dim)\n")

        print(f"  image model: {settings.image_embedding_model} ({settings.image_embedding_pretrained})")
        from core.embeddings.image_embedding_service import ImageEmbeddingService

        dim = len(ImageEmbeddingService().encode_text("hello"))
        print(f"               ready ({dim}-dim)\n")
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nCheck the internet connection and run this again. Nothing is "
            "half-installed - the download resumes from where it stopped.",
            file=sys.stderr,
        )
        return 1

    after = _cache_size_mb()
    downloaded = after - before
    print(f"Model cache: {after:.0f} MB", end="")
    print(f" ({downloaded:.0f} MB downloaded)" if downloaded > 1 else " (already present)")
    print("\nSetup complete. OmniFind will now run without an internet connection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
