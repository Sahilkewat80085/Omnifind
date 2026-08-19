from pathlib import Path

import pytest

from core.embeddings.text_embedding_service import TextEmbeddingService
from core.llm.base import LLMNotConfiguredError, LLMProvider
from core.llm.gemini_service import GeminiService
from core.vectorstore.qdrant_client import SearchHit, VectorService
from services.rag_service import NO_CONTEXT_ANSWER, NO_TEXT_CONTEXT_ANSWER, RagService


class FakeLLM:
    """Records what it was asked so tests can assert on the prompt itself."""

    def __init__(self, reply: str = "The fee paid was 15,000 [1].") -> None:
        self._reply = reply
        self.calls: list[dict[str, str]] = []

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "fake-model"

    def generate(self, *, system_instruction: str, prompt: str) -> str:
        self.calls.append({"system_instruction": system_instruction, "prompt": prompt})
        return self._reply


class FakeVectorService:
    def __init__(self, hits: list[SearchHit], image_hits: list[SearchHit] | None = None) -> None:
        self._hits = hits
        self._image_hits = image_hits or []

    def search_text(self, query_vector: list[float], top_k: int) -> list[SearchHit]:
        return self._hits[:top_k]

    def search_image(self, query_vector: list[float], top_k: int) -> list[SearchHit]:
        return self._image_hits[:top_k]


class FakeEmbedder:
    def encode_query(self, text: str) -> list[float]:
        return [0.0] * 384


class FakeImageEmbedder:
    def encode_text(self, text: str) -> list[float]:
        return [0.0] * 512


class ExplodingImageEmbedder:
    def encode_text(self, text: str) -> list[float]:
        raise RuntimeError("open_clip is not installed")


_fake_root: Path


@pytest.fixture(autouse=True, scope="module")
def _fake_file_root(tmp_path_factory):
    """Give the fabricated hits paths that really exist on disk.

    Retrieval drops any hit whose file is gone, so that a file deleted after
    indexing can never be cited. Hits pointing at an invented path would be
    filtered out before any of these assertions ran.
    """
    global _fake_root
    _fake_root = tmp_path_factory.mktemp("indexed_files")
    yield


def _fake_path(name: str) -> str:
    path = _fake_root / name
    path.touch()
    return str(path)


def _hit(score: float, name: str = "fees.pdf", text: str = "Total fee paid: 15,000.") -> SearchHit:
    return SearchHit(
        score=score,
        payload={
            "file_id": "f1",
            "file_name": name,
            "path": _fake_path(name),
            "page_number": 2,
            "chunk_text": text,
            "chunk_index": 0,
        },
    )


def _image_hit(score: float, name: str = "beach.jpg") -> SearchHit:
    """A raw CLIP hit, pre-calibration — scores here are on CLIP's own scale
    (~0.05 noise, ~0.22-0.33 for a real match), not the 0-1 UI scale."""
    return SearchHit(
        score=score,
        payload={
            "file_id": "i1",
            "file_name": name,
            "path": _fake_path(name),
            "file_type": "image",
            "image_dimensions": {"width": 800, "height": 600},
        },
    )


def test_fake_llm_satisfies_the_provider_protocol():
    # Guards the seam RagService depends on: if LLMProvider grows a member,
    # this fails here rather than silently in every other test's fake.
    assert isinstance(FakeLLM(), LLMProvider)
    assert isinstance(GeminiService(), LLMProvider)


def test_ask_numbers_excerpts_and_returns_matching_citations():
    llm = FakeLLM()
    service = RagService(
        llm=llm,
        vector_service=FakeVectorService([_hit(0.9), _hit(0.8, name="invoice.pdf", text="Amount due: 900.")]),
        text_embedder=FakeEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )

    response = service.ask("how much did I pay in fees?")

    assert response.used_context is True
    assert response.model == "fake-model"
    assert [c.marker for c in response.citations] == [1, 2]
    assert [c.file_name for c in response.citations] == ["fees.pdf", "invoice.pdf"]

    prompt = llm.calls[0]["prompt"]
    # The markers the model is told to cite must line up with the citation
    # markers the UI renders, or [1] in the answer points at the wrong file.
    assert '[1] From "fees.pdf", page 2:' in prompt
    assert '[2] From "invoice.pdf", page 2:' in prompt
    assert "Total fee paid: 15,000." in prompt
    assert "how much did I pay in fees?" in prompt


def test_ask_without_relevant_context_does_not_call_the_model():
    llm = FakeLLM()
    service = RagService(
        llm=llm,
        vector_service=FakeVectorService([]),
        text_embedder=FakeEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )

    response = service.ask("anything at all")

    assert response.used_context is False
    assert response.answer == NO_CONTEXT_ANSWER
    assert response.citations == []
    # Spending an API call to answer "I have no documents" would be pure waste.
    assert llm.calls == []


def test_hits_below_the_relevance_floor_are_dropped(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SIMILARITY", "0.5")
    from utils.config import get_settings

    get_settings.cache_clear()
    try:
        service = RagService(
            llm=FakeLLM(),
            vector_service=FakeVectorService([_hit(0.9), _hit(0.2, name="unrelated.pdf")]),
            text_embedder=FakeEmbedder(),
            image_embedder=FakeImageEmbedder(),
        )
        response = service.ask("fees")

        assert [c.file_name for c in response.citations] == ["fees.pdf"]
    finally:
        get_settings.cache_clear()


def test_top_k_limits_how_many_chunks_reach_the_prompt():
    llm = FakeLLM()
    service = RagService(
        llm=llm,
        vector_service=FakeVectorService([_hit(0.9 - i / 100, name=f"doc{i}.pdf") for i in range(10)]),
        text_embedder=FakeEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )

    response = service.ask("fees", top_k=3)

    assert len(response.citations) == 3
    assert "doc3.pdf" not in llm.calls[0]["prompt"]


def test_ask_end_to_end_over_real_embeddings_and_vector_store(isolated_env):
    vector_service = VectorService()
    vector_service.ensure_collection()
    embedder = TextEmbeddingService()

    passages = [
        ("dbms_notes.pdf", "Database normalization removes redundant data from tables."),
        ("trip.pdf", "The brochure lists beach resorts in Goa and Kerala."),
    ]
    for name, text in passages:
        vector_service.upsert_text_chunk(
            file_id=name,
            file_name=name,
            path=_fake_path(name),
            page_number=1,
            chunk_text=text,
            chunk_index=0,
            vector=embedder.encode_documents([text])[0],
        )

    llm = FakeLLM(reply="Normalization removes redundancy [1].")
    response = RagService(
        llm=llm,
        vector_service=vector_service,
        text_embedder=embedder,
        image_embedder=FakeImageEmbedder(),
    ).ask(
        "what does normalizing a database achieve?", top_k=1
    )

    assert response.used_context is True
    # Semantic retrieval, not keyword: the question shares no distinctive word
    # with the passage it must find.
    assert response.citations[0].file_name == "dbms_notes.pdf"
    assert "redundant data" in llm.calls[0]["prompt"]


def test_a_deleted_file_is_never_cited(isolated_env):
    """A file deleted after indexing must not answer a question.

    Its vectors survive until the folder is re-scanned, so without the disk
    check the model would be handed an excerpt from a file that is gone and
    would cite it — a source the user cannot open.
    """
    live = _hit(0.9, name="fees.pdf", text="Total fee paid: 15,000.")
    deleted = _hit(0.95, name="old_fees.pdf", text="Total fee paid: 99,999.")
    Path(deleted.payload["path"]).unlink()

    llm = FakeLLM()
    response = RagService(
        llm=llm,
        vector_service=FakeVectorService([deleted, live]),
        text_embedder=FakeEmbedder(),
        image_embedder=FakeImageEmbedder(),
    ).ask("what was the fee?")

    assert [c.file_name for c in response.citations] == ["fees.pdf"]
    assert "99,999" not in llm.calls[0]["prompt"]


def test_a_deleted_image_is_not_offered_alongside_an_answer(isolated_env):
    kept = _image_hit(0.30, "invoice_scan.jpg")
    deleted = _image_hit(0.31, "old_scan.jpg")
    Path(deleted.payload["path"]).unlink()

    response = RagService(
        llm=FakeLLM(),
        vector_service=FakeVectorService([_hit(0.9)], image_hits=[deleted, kept]),
        text_embedder=FakeEmbedder(),
        image_embedder=FakeImageEmbedder(),
    ).ask("what was the fee?")

    assert [i.file_name for i in response.related_images] == ["invoice_scan.jpg"]


def test_gemini_service_without_api_key_reports_not_configured(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    from utils.config import get_settings

    get_settings.cache_clear()
    try:
        service = GeminiService()
        assert service.is_configured is False
        with pytest.raises(LLMNotConfiguredError):
            service.generate(system_instruction="s", prompt="p")
    finally:
        get_settings.cache_clear()


# ---------- related images on the Ask page ----------
#
# Images are retrieved on a separate pass and never enter the prompt. These
# lock in that split: the user sees the pictures, the model does not.


def test_matching_images_are_returned_but_never_reach_the_prompt():
    llm = FakeLLM()
    service = RagService(
        llm=llm,
        vector_service=FakeVectorService([_hit(0.9)], image_hits=[_image_hit(0.30)]),
        text_embedder=FakeEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )

    response = service.ask("what did the beach trip cost?")

    assert [img.file_name for img in response.related_images] == ["beach.jpg"]
    assert response.related_images[0].width == 800
    # The whole point of the separation: an image the model cannot read must
    # not appear as grounding it was asked to cite.
    assert "beach.jpg" not in llm.calls[0]["prompt"]
    assert all(c.file_name != "beach.jpg" for c in response.citations)


def test_related_images_use_the_same_calibration_as_the_search_page():
    """A 0.30 raw CLIP score must land at the same percentage on both tabs."""
    service = RagService(
        llm=FakeLLM(),
        vector_service=FakeVectorService([_hit(0.9)], image_hits=[_image_hit(0.30)]),
        text_embedder=FakeEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )

    response = service.ask("beach")

    # (0.30 - 0.18) / (0.35 - 0.18) = 0.7058…
    assert response.related_images[0].similarity == pytest.approx(0.70588, abs=1e-4)


def test_sub_noise_images_are_dropped_from_an_answer():
    """An unrelated picture beside a cited answer reads as a wrong answer."""
    service = RagService(
        llm=FakeLLM(),
        vector_service=FakeVectorService(
            [_hit(0.9)],
            image_hits=[_image_hit(0.31, "invoice_scan.jpg"), _image_hit(0.06, "cat.jpg")],
        ),
        text_embedder=FakeEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )

    response = service.ask("what was billed?")

    assert [img.file_name for img in response.related_images] == ["invoice_scan.jpg"]


def test_images_still_show_when_no_document_matched():
    """A purely visual question finds no text, and must not claim it found nothing."""
    llm = FakeLLM()
    service = RagService(
        llm=llm,
        vector_service=FakeVectorService([], image_hits=[_image_hit(0.29, "sunset.png")]),
        text_embedder=FakeEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )

    response = service.ask("a photo of a sunset")

    assert response.used_context is False
    assert [img.file_name for img in response.related_images] == ["sunset.png"]
    assert response.answer == NO_TEXT_CONTEXT_ANSWER
    assert response.answer != NO_CONTEXT_ANSWER
    # Still no reason to spend an API call — there is nothing to quote.
    assert llm.calls == []


def test_broken_image_retrieval_does_not_break_the_answer():
    """No OpenCLIP must degrade /ask to text-only, not 500 it."""
    service = RagService(
        llm=FakeLLM(),
        vector_service=FakeVectorService([_hit(0.9)], image_hits=[_image_hit(0.30)]),
        text_embedder=FakeEmbedder(),
        image_embedder=ExplodingImageEmbedder(),
    )

    response = service.ask("how much did I pay in fees?")

    assert response.used_context is True
    assert response.citations[0].file_name == "fees.pdf"
    assert response.related_images == []
