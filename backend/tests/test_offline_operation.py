"""OmniFind needs the internet once, at setup, and never again.

The rule these lock in: the app must load its embedding models from the local
cache and never reach for huggingface.co at runtime. The subtle part is
ordering - huggingface_hub copies HF_HUB_OFFLINE into a module constant when
it is imported, so enforcing offline mode a moment too late does nothing and
fails silently, looking exactly like success until someone runs it on a train.
"""

import os
import subprocess
import sys
from pathlib import Path

from core.embeddings.errors import ModelsNotAvailableError
from utils.offline import enforce_offline_models, model_downloads_allowed

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def test_enforce_offline_sets_the_hub_variables(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("OMNIFIND_ALLOW_MODEL_DOWNLOAD", raising=False)

    enforce_offline_models()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_an_explicit_override_is_respected(monkeypatch):
    """setdefault, not assignment - debugging a model problem stays possible."""
    monkeypatch.delenv("OMNIFIND_ALLOW_MODEL_DOWNLOAD", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    enforce_offline_models()

    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_the_fetch_script_is_allowed_online(monkeypatch):
    """Otherwise setup could never download anything in the first place."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("OMNIFIND_ALLOW_MODEL_DOWNLOAD", "1")

    assert model_downloads_allowed() is True
    enforce_offline_models()

    assert "HF_HUB_OFFLINE" not in os.environ


def test_importing_main_puts_huggingface_hub_in_offline_mode():
    """The ordering guarantee, checked the only way that proves it.

    Asserting on utils/offline.py alone would pass even if main.py imported a
    model library first and rendered the whole thing inert. This imports the
    real app in a clean interpreter and asks huggingface_hub what it believes,
    which is the thing that actually decides whether a request goes out.
    """
    probe = (
        "import main, huggingface_hub.constants as c; "
        "print('OFFLINE=' + str(c.HF_HUB_OFFLINE))"
    )
    env = {**os.environ}
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    assert "OFFLINE=True" in result.stdout, (
        f"huggingface_hub was imported before offline mode was enforced.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )


def test_missing_models_raise_a_message_that_names_the_fix():
    """The HuggingFace OSError says "check your internet connection", which is
    the one thing that will not help a user who is meant to be offline."""
    error = ModelsNotAvailableError("BAAI/bge-small-en-v1.5", OSError("We couldn't connect"))
    message = str(error)

    assert "scripts/fetch_models.py" in message
    assert "runs fully offline" in message
    # ASCII only: this reaches a cp1252 Windows console, which turns anything
    # fancier into a question mark.
    message.encode("ascii")


def test_the_fetch_script_allows_downloads_before_importing_anything():
    """A one-line ordering bug in the script would leave setup unable to
    download, and the failure would look like a network fault."""
    source = (BACKEND_ROOT / "scripts" / "fetch_models.py").read_text(encoding="utf-8")

    allow_at = source.index("OMNIFIND_ALLOW_MODEL_DOWNLOAD")
    first_app_import = source.index("from utils.config import get_settings")

    assert allow_at < first_app_import
