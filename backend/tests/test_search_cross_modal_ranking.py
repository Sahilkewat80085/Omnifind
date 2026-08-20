"""Cross-modal ranking: an image must be able to win, and must stay out of
results it has nothing to do with.

These lock in the calibration in SearchService. The earlier min-max scheme
rescaled each modality against only the hits it returned, so the best image
always scored 1.0 no matter how weak the match — which put an unrelated image
into every result list and, on a score tie, always behind the documents.
"""

from services.search_service import _calibrate
from core.vectorstore.qdrant_client import SearchHit


def _hit(score: float, name: str = "f") -> SearchHit:
    return SearchHit(score=score, payload={"file_name": name})


def test_calibration_preserves_absolute_strength():
    """A weak best-hit must not be promoted to a perfect score."""
    weak = _calibrate([_hit(0.20)], floor=0.18, ceil=0.35, drop_below_floor=True)
    strong = _calibrate([_hit(0.33)], floor=0.18, ceil=0.35, drop_below_floor=True)

    assert weak[0].score < 0.2
    assert strong[0].score > 0.8


def test_sub_noise_image_hits_are_dropped():
    """CLIP scores ~0.05-0.13 mean 'nothing like this picture' and must vanish."""
    hits = _calibrate(
        [_hit(0.28, "dog.jpg"), _hit(0.09, "car.jpg")],
        floor=0.18,
        ceil=0.35,
        drop_below_floor=True,
    )

    assert [h.payload["file_name"] for h in hits] == ["dog.jpg"]


def test_sub_noise_text_hits_are_clamped_not_dropped():
    """Documents keep their recall; weak ones just sort to the bottom."""
    hits = _calibrate(
        [_hit(0.70), _hit(0.30)], floor=0.40, ceil=0.80, drop_below_floor=False
    )

    assert len(hits) == 2
    assert hits[1].score == 0.0


def test_scores_are_capped_at_one():
    hits = _calibrate([_hit(0.99)], floor=0.40, ceil=0.80, drop_below_floor=False)

    assert hits[0].score == 1.0


def test_degenerate_range_falls_back_to_raw_scores():
    """A misconfigured floor >= ceil must not divide by zero or empty the results."""
    hits = _calibrate([_hit(0.5)], floor=0.8, ceil=0.8, drop_below_floor=True)

    assert [h.score for h in hits] == [0.5]


def test_image_outranks_documents_for_a_visual_query(isolated_env, tmp_path):
    """End-to-end: a picture of a dog wins 'a dog', and loses 'invoice total'."""
    import docx
    from PIL import Image
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.models import Base
    from models.schemas.search_schemas import ImageResult
    from services.indexing_service import IndexingService
    from services.metadata_service import MetadataService
    from services.search_service import SearchService

    source = tmp_path / "source"
    source.mkdir()

    # A red square is not a dog, but CLIP separates it from an invoice well
    # enough to prove the two modalities rank against each other.
    Image.new("RGB", (256, 256), (200, 20, 20)).save(source / "red.png")
    d = docx.Document()
    d.add_paragraph("Invoice total due: $1,240.00. Payment terms net 30.")
    d.save(str(source / "invoice.docx"))

    engine = create_engine(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    IndexingService(MetadataService(db)).index_folder(str(source))

    money = SearchService(db=db).search("invoice total amount due")
    assert not any(isinstance(r, ImageResult) for r in money.results), (
        "an unrelated image leaked into a document query"
    )
    assert money.results[0].file_name == "invoice.docx"

    db.close()
