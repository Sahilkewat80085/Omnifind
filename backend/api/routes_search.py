from fastapi import APIRouter, Query

from models.schemas.file_schemas import FileType
from models.schemas.search_schemas import SearchResponse
from services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    # Counts files, not chunks. The UI asks for more than it puts on screen so
    # its "explore more" control can reveal the rest without a second search.
    limit: int | None = Query(None, ge=1, le=100),
    # Set by the type filter on the Search page. Overrides any type word read
    # out of the query itself — see SearchService.search.
    file_type: FileType | None = Query(None),
) -> SearchResponse:
    return SearchService().search(q, top_k=limit, file_type=file_type)
