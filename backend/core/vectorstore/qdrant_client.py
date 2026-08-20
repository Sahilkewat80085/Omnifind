import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from utils.config import BACKEND_ROOT, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

TEXT_VECTOR_NAME = "text_vector"
IMAGE_VECTOR_NAME = "image_vector"
TEXT_VECTOR_SIZE = 384
IMAGE_VECTOR_SIZE = 512


@dataclass(frozen=True)
class SearchHit:
    score: float
    payload: dict[str, object]


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_mode == "local":
        local_path = Path(settings.qdrant_local_path)
        if not local_path.is_absolute():
            local_path = (BACKEND_ROOT / local_path).resolve()
        local_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(local_path))
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


class VectorService:
    def __init__(self) -> None:
        self._client = _get_client()
        self._collection = get_settings().qdrant_collection_name

    def collection_exists(self) -> bool:
        """False until the first folder is indexed.

        The collection is created by indexing, not at startup, so on a fresh
        install it genuinely does not exist yet - and Qdrant raises rather
        than returning nothing when you search a collection that is not there.
        """
        return self._collection in {
            c.name for c in self._client.get_collections().collections
        }

    def ensure_collection(self) -> None:
        if self.collection_exists():
            return

        logger.info("Creating Qdrant collection: %s", self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                TEXT_VECTOR_NAME: qmodels.VectorParams(
                    size=TEXT_VECTOR_SIZE,
                    distance=qmodels.Distance.COSINE,
                ),
                IMAGE_VECTOR_NAME: qmodels.VectorParams(
                    size=IMAGE_VECTOR_SIZE,
                    distance=qmodels.Distance.COSINE,
                ),
            },
        )

    def upsert_text_chunk(
        self,
        *,
        file_id: str,
        file_name: str,
        path: str,
        page_number: int | None,
        chunk_text: str,
        chunk_index: int,
        vector: list[float],
    ) -> None:
        point = qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector={TEXT_VECTOR_NAME: vector},
            payload={
                "file_id": file_id,
                "file_name": file_name,
                "file_type": "document",
                "path": path,
                "page_number": page_number,
                "chunk_text": chunk_text,
                "chunk_index": chunk_index,
            },
        )
        self._client.upsert(collection_name=self._collection, points=[point])

    def upsert_code_chunk(
        self,
        *,
        file_id: str,
        file_name: str,
        path: str,
        language: str,
        symbol: str | None,
        line_start: int,
        line_end: int,
        chunk_text: str,
        chunk_index: int,
        vector: list[float],
    ) -> None:
        point = qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector={TEXT_VECTOR_NAME: vector},
            payload={
                "file_id": file_id,
                "file_name": file_name,
                "file_type": "code",
                "path": path,
                "language": language,
                "symbol": symbol,
                "line_start": line_start,
                "line_end": line_end,
                "chunk_text": chunk_text,
                "chunk_index": chunk_index,
            },
        )
        self._client.upsert(collection_name=self._collection, points=[point])

    def upsert_image(
        self,
        *,
        file_id: str,
        file_name: str,
        path: str,
        width: int,
        height: int,
        vector: list[float],
    ) -> None:
        point = qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector={IMAGE_VECTOR_NAME: vector},
            payload={
                "file_id": file_id,
                "file_name": file_name,
                "file_type": "image",
                "path": path,
                "image_dimensions": {"width": width, "height": height},
            },
        )
        self._client.upsert(collection_name=self._collection, points=[point])

    def upsert_points(self, points: list[qmodels.PointStruct]) -> None:
        """Upserts a batch of points (vectors and payloads) into the collection."""
        if not points:
            return
        self.ensure_collection()
        self._client.upsert(collection_name=self._collection, points=points)

    def search_text(
        self,
        query_vector: list[float],
        top_k: int,
        file_type: str | None = None,
    ) -> list[SearchHit]:

        # Nothing indexed yet means no hits, not an error. Without this the
        # first thing a new user does - install, open the app, type a query -
        # is answered with a 500, because searching a collection that does not
        # exist raises inside qdrant-client.
        if not self.collection_exists():
            return []

        query_filter = None
        if file_type is not None:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="file_type", match=qmodels.MatchValue(value=file_type)
                    )
                ]
            )
        results = self._client.search(
            collection_name=self._collection,
            query_vector=(TEXT_VECTOR_NAME, query_vector),
            query_filter=query_filter,
            limit=top_k,
        )
        return [SearchHit(score=r.score, payload=r.payload or {}) for r in results]

    def search_image(self, query_vector: list[float], top_k: int) -> list[SearchHit]:
        if not self.collection_exists():
            return []

        results = self._client.search(
            collection_name=self._collection,
            query_vector=(IMAGE_VECTOR_NAME, query_vector),
            limit=top_k,
        )
        return [SearchHit(score=r.score, payload=r.payload or {}) for r in results]

    def iter_payloads(self, batch: int = 512) -> Iterator[dict[str, Any]]:
        """Walk every indexed point's payload.

        Exists so an index built before chunk text was mirrored into SQLite can
        be repaired from what Qdrant already holds, instead of re-embedding
        every file to recover text that was never lost.
        """
        if not self.collection_exists():
            return

        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                with_payload=True,
                with_vectors=False,
                limit=batch,
                offset=offset,
            )
            for point in points:
                if point.payload:
                    yield point.payload
            if offset is None:
                return

    def get_points_by_path(self, path: str) -> list[dict[str, Any]]:
        """Retrieve all indexed points and vectors for a given file path."""
        try:
            scroll_result, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="path", match=qmodels.MatchValue(value=path))]
                ),
                with_payload=True,
                with_vectors=True,
                limit=100,
            )
        except Exception:
            logger.exception("Failed to scroll points for path: %s", path)
            return []

        points: list[dict[str, Any]] = []
        for p in scroll_result:
            vectors_info: dict[str, Any] = {}
            if isinstance(p.vector, dict):
                for v_name, v_val in p.vector.items():
                    if isinstance(v_val, list):
                        vectors_info[v_name] = {
                            "dimensions": len(v_val),
                            "sample": [round(float(x), 5) for x in v_val[:8]],
                        }
            elif isinstance(p.vector, list):
                vectors_info["default"] = {
                    "dimensions": len(p.vector),
                    "sample": [round(float(x), 5) for x in p.vector[:8]],
                }
            points.append({
                "id": str(p.id),
                "payload": p.payload or {},
                "vectors": vectors_info,
            })
        return points

    def delete_by_path(self, path: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="path", match=qmodels.MatchValue(value=path))]
                )
            ),
        )
