from fastapi import APIRouter, Query

from models.schemas.file_schemas import FileType
from models.schemas.search_schemas import SearchResponse
from services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    limit: int | None = Query(None, ge=1, le=500),
    file_type: FileType | None = Query(None),
) -> SearchResponse:
    return SearchService().search(q, top_k=limit, file_type=file_type)

