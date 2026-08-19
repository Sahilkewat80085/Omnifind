from fastapi import APIRouter, HTTPException

from core.llm.base import LLMError, LLMNotConfiguredError
from models.schemas.rag_schemas import AskRequest, AskResponse
from services.rag_service import RagService

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return RagService().ask(request.q, top_k=request.top_k)
    except LLMNotConfiguredError as exc:
        # 503, not 500: the engine is fine, this install just has no key yet.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
