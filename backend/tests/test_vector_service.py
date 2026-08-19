from core.vectorstore.qdrant_client import VectorService


def test_named_vector_partitions_and_delete_by_path(isolated_env):
    svc = VectorService()
    svc.ensure_collection()
    svc.ensure_collection()  # must be idempotent

    text_vec = [0.01] * 384
    image_vec = [0.02] * 512

    svc.upsert_text_chunk(
        file_id="f1", file_name="notes.txt", path="C:/fake/notes.txt",
        page_number=1, chunk_text="hello world", chunk_index=0, vector=text_vec,
    )
    svc.upsert_image(
        file_id="f2", file_name="pic.png", path="C:/fake/pic.png",
        width=800, height=600, vector=image_vec,
    )

    text_hits = svc.search_text(text_vec, top_k=5)
    image_hits = svc.search_image(image_vec, top_k=5)

    assert len(text_hits) == 1
    assert text_hits[0].payload["file_type"] == "document"
    assert len(image_hits) == 1
    assert image_hits[0].payload["file_type"] == "image"

    svc.delete_by_path("C:/fake/notes.txt")
    assert svc.search_text(text_vec, top_k=5) == []
    # deleting one path must not affect unrelated points
    assert len(svc.search_image(image_vec, top_k=5)) == 1
