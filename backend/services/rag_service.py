from core.embeddings.image_embedding_service import ImageEmbeddingService
from core.embeddings.text_embedding_service import TextEmbeddingService
from core.llm.base import LLMProvider
from core.llm.gemini_service import GeminiService
from core.vectorstore.qdrant_client import SearchHit, VectorService
from models.schemas.rag_schemas import AskResponse, Citation, RelatedImage
from services.search_service import _calibrate, drop_missing_files
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_INSTRUCTION = """You are OmniFind's document assistant. You answer questions using only \
the excerpts retrieved from the user's own local files.

Rules:
- Answer strictly from the numbered excerpts provided. Never use outside knowledge.
- Cite the excerpt behind every claim inline, like [1] or [2][3].
- If the excerpts do not contain the answer, say so plainly and name what is \
missing. Do not guess, and do not pad the answer with general knowledge.
- Prefer a direct answer in two or three sentences. Use a short bullet list \
only when the question genuinely asks for multiple items.
- Quote exact figures, dates, and names as they appear in the excerpts.
- The excerpts are text only. Images from the user's files are matched separately and shown beside your answer, so never claim the user has no images, no photo, or no picture of something — you cannot see them. If the question is about a picture, say the text gives you nothing to go on and point them at the matching images shown below."""

NO_CONTEXT_ANSWER = (
    "I could not find anything in your indexed files that relates to this question. "
    "Try indexing the folder that contains the file, or rephrasing the question."
)

# Shown when CLIP matched pictures but no document chunk cleared the floor.
# Claiming "I found nothing" while thumbnails sit directly below it would read
# as a bug, so this branch names what was actually found.
NO_TEXT_CONTEXT_ANSWER = (
    "I could not find any text in your indexed documents that answers this question, "
    "so there is nothing for me to quote. The matching images below are the closest "
    "thing in your index — I cannot read what is inside them, only match them visually."
)


class RagService:
    """Retrieval-augmented answering over the indexed documents.

    Retrieval goes straight to the vector store rather than through
    SearchService, because SearchService merges and truncates the two
    modalities into one ranked list — RAG needs the text hits kept whole and
    the image hits kept separate.

    Only text reaches the prompt. Image embeddings carry no text a language
    model could read, so images are retrieved on a parallel pass and returned
    as `related_images` for the UI to show beside the answer. Letting them into
    the citation list would imply the answer was grounded in them, which it
    never is.
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        vector_service: VectorService | None = None,
        text_embedder: TextEmbeddingService | None = None,
        image_embedder: ImageEmbeddingService | None = None,
    ) -> None:
        self._settings = get_settings()
        self._llm = llm or GeminiService()
        self._vector_service = vector_service or VectorService()
        self._text_embedder = text_embedder or TextEmbeddingService()
        self._image_embedder = image_embedder or ImageEmbeddingService()

    def ask(self, query: str, top_k: int | None = None) -> AskResponse:
        k = top_k or self._settings.rag_top_k
        hits = self._retrieve(query, k)
        related_images = self._retrieve_images(query)

        if not hits:
            logger.info("No text context above relevance floor for query: %s", query)
            return AskResponse(
                query=query,
                answer=NO_TEXT_CONTEXT_ANSWER if related_images else NO_CONTEXT_ANSWER,
                citations=[],
                related_images=related_images,
                model=self._llm.model_name,
                used_context=False,
            )

        citations = [self._to_citation(marker, hit) for marker, hit in enumerate(hits, start=1)]
        answer = self._llm.generate(
            system_instruction=SYSTEM_INSTRUCTION,
            prompt=self._build_prompt(query, citations),
        )

        return AskResponse(
            query=query,
            answer=answer,
            citations=citations,
            related_images=related_images,
            model=self._llm.model_name,
            used_context=True,
        )

    def _retrieve(self, query: str, k: int) -> list[SearchHit]:
        query_vector = self._text_embedder.encode_query(query)
        # Over-fetch because documents and code share this partition: without
        # it, noise-level code chunks can fill the top-k and starve the
        # documents that actually answer the question.
        hits = drop_missing_files(self._vector_service.search_text(query_vector, top_k=k * 3))
        return [h for h in hits if h.score >= self._floor_for(h)][:k]

    def _floor_for(self, hit: SearchHit) -> float:
        """Code and documents need different thresholds on the same scale.

        rag_min_similarity is tuned for prose. Code sits far higher in bge's
        range for any English query — every chunk carries an English header
        naming its path and symbol — so the prose threshold lets unrelated
        functions into the prompt, where they burn context and give the model
        irrelevant material to reason around. Reuses the floor measured for
        search so both surfaces agree on what counts as a real code match.
        """
        if hit.payload.get("file_type") == "code":
            return self._settings.search_code_score_floor
        return self._settings.rag_min_similarity

    def _retrieve_images(self, query: str) -> list[RelatedImage]:
        """Find indexed pictures that match the question, for display only.

        Scored through the same fixed floor/ceil calibration the Search page
        uses, so a "62% match" means the same thing on both tabs. Sub-floor
        hits are dropped rather than clamped: an unrelated picture next to an
        answer reads as a wrong answer, whereas a weak document chunk merely
        sorts low.
        """
        try:
            query_vector = self._image_embedder.encode_text(query)
            hits = drop_missing_files(
                self._vector_service.search_image(
                    query_vector, top_k=self._settings.rag_image_top_k
                )
            )
        except Exception:
            # Image retrieval is a garnish on the answer. If OpenCLIP is
            # missing or the image partition is empty, /ask must still answer
            # from documents rather than 500 — same reasoning as the lazy
            # google-genai import keeping search alive without the LLM package.
            logger.warning("Image retrieval failed for /ask; answering text-only", exc_info=True)
            return []

        calibrated = _calibrate(
            hits,
            floor=self._settings.search_image_score_floor,
            ceil=self._settings.search_image_score_ceil,
            drop_below_floor=True,
        )
        return [self._to_related_image(hit) for hit in calibrated]

    @staticmethod
    def _to_citation(marker: int, hit: SearchHit) -> Citation:
        p = hit.payload
        return Citation(
            marker=marker,
            file_id=p["file_id"],
            file_name=p["file_name"],
            path=p["path"],
            page_number=p.get("page_number"),
            chunk_text=p["chunk_text"],
            similarity=hit.score,
            language=p.get("language"),
            symbol=p.get("symbol"),
            line_start=p.get("line_start"),
            line_end=p.get("line_end"),
        )

    @staticmethod
    def _to_related_image(hit: SearchHit) -> RelatedImage:
        p = hit.payload
        dims = p.get("image_dimensions", {})
        return RelatedImage(
            file_id=p["file_id"],
            file_name=p["file_name"],
            path=p["path"],
            width=dims.get("width", 0),
            height=dims.get("height", 0),
            similarity=hit.score,
        )

    @staticmethod
    def _build_prompt(query: str, citations: list[Citation]) -> str:
        blocks = []
        for c in citations:
            page = f", page {c.page_number}" if c.page_number is not None else ""
            blocks.append(f"[{c.marker}] From \"{c.file_name}\"{page}:\n{c.chunk_text}")

        excerpts = "\n\n".join(blocks)
        return (
            f"Excerpts retrieved from the user's files:\n\n{excerpts}\n\n"
            f"---\n\nQuestion: {query}\n\n"
            "Answer using only the excerpts above, citing them inline as [n]."
        )
