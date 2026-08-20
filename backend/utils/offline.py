"""Pin model loading to what is already on disk.

OmniFind needs the internet exactly once — at install time, to download the
two embedding models. After that it must never reach for the network again:
a desktop search tool that stalls because huggingface.co is unreachable is
broken, and the user may well be on a train.

**Import order matters here.** `huggingface_hub` copies HF_HUB_OFFLINE into a
module constant when it is first imported, so setting the variable afterwards
is silently a no-op — verified: the constant stays False. That is why
`enforce_offline_models()` has to run before anything pulls in
sentence-transformers, open_clip or transformers, and why `main.py` calls it
above its own imports.

`scripts/fetch_models.py` is the one caller that must be allowed out: it sets
OMNIFIND_ALLOW_MODEL_DOWNLOAD=1 so this becomes a no-op for that process.
"""

import os

ALLOW_DOWNLOAD_ENV = "OMNIFIND_ALLOW_MODEL_DOWNLOAD"


def model_downloads_allowed() -> bool:
    """True only for the install-time fetch script."""
    return os.environ.get(ALLOW_DOWNLOAD_ENV, "").strip() == "1"


def enforce_offline_models() -> None:
    """Make the HuggingFace libraries load from the local cache only.

    `setdefault`, not assignment: someone debugging a model problem can still
    export HF_HUB_OFFLINE=0 and have it respected.
    """
    if model_downloads_allowed():
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
