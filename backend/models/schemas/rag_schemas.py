from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    q: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    """One retrieved chunk that was actually put in front of the model.

    `marker` is the number the answer cites inline as [1], [2], … so the UI
    can link a sentence back to the exact page it came from.
    """

    marker: int
    file_id: str
    file_name: str
    path: str
    page_number: int | None
    chunk_text: str
    similarity: float
    # Set for source-code excerpts, which have lines where a document has a
    # page. Both are None for a plain .txt, which has neither.
    language: str | None = None
    symbol: str | None = None
    line_start: int | None = None
    line_end: int | None = None


class RelatedImage(BaseModel):
    """An indexed image that matches the question, shown beside the answer.

    Deliberately not a Citation: images are retrieved by CLIP and never reach
    the prompt, so they carry no `marker` — nothing in the answer text can
    cite them. They are visual context for the user, not grounding for the
    model, and the UI must keep that distinction visible.
    """

    file_id: str
    file_name: str
    path: str
    width: int
    height: int
    similarity: float


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    related_images: list[RelatedImage] = []
    model: str
    # False when retrieval found nothing above the relevance floor, so the
    # answer is a canned "nothing indexed matches" rather than model output.
    # The UI uses this to avoid rendering an empty sources panel. Note this
    # tracks *text* context only — related_images can be non-empty while this
    # is False, when a question matches pictures but no documents.
    used_context: bool
