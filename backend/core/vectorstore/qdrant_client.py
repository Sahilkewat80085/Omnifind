import os
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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

    def ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
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

    def search_text(
        self,
        query_vector: list[float],
        top_k: int,
        file_type: str | None = None,
    ) -> list[SearchHit]:
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
        results = self._client.search(
            collection_name=self._collection,
            query_vector=(IMAGE_VECTOR_NAME, query_vector),
            limit=top_k,
        )
        return [SearchHit(score=r.score, payload=r.payload or {}) for r in results]

    def delete_by_path(self, path: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="path", match=qmodels.MatchValue(value=path))]
                )
            ),
        )
