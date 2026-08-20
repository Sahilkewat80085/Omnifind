from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "OmniFind"
    app_env: str = "development"
    log_level: str = "INFO"

    # SQLite
    database_url: str = "sqlite:///./storage/omnifind.db"

    # Qdrant — "local" runs embedded/on-disk with no server process (default,
    # fits a desktop app); "server" connects to a Qdrant instance you run yourself.
    qdrant_mode: str = "local"
    qdrant_local_path: str = "./storage/qdrant_local"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "omnifind_assets"

    # Embedding models
    text_embedding_model: str = "BAAI/bge-small-en-v1.5"
    image_embedding_model: str = "ViT-B-32"
    image_embedding_pretrained: str = "laion2b_s34b_b79k"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Code indexing (Milestone 3)
    #
    # Code is chunked by LINES, not words: prose chunking joins on whitespace,
    # which erases the indentation and line breaks that carry a program's
    # structure. Chunks follow function/class boundaries where the language
    # makes them findable, so a hit is a whole symbol rather than an arbitrary
    # window through the middle of one.
    code_chunk_max_lines: int = 80
    code_chunk_overlap_lines: int = 10
    # Blocks below this merge into their neighbour — a lone one-line getter is
    # noise on its own, and dilutes the ranking with near-empty vectors.
    code_chunk_min_lines: int = 4
    # Skips minified bundles, generated parsers and vendored blobs, which are
    # technically source but never what anyone is searching for.
    code_max_file_bytes: int = 1_000_000

    # Search
    search_top_k: int = 10

    # Cross-modal score calibration.
    #
    # bge and CLIP cosine scores are not comparable: bge sits around 0.40-0.50
    # even for a query with nothing to do with the corpus, and reaches ~0.80 on
    # a strong hit, while CLIP text-image similarity is ~0.05 for noise and only
    # ~0.22-0.33 when the image genuinely matches. Ranking the two together on
    # raw cosine buries every image; min-max normalizing each modality (the
    # previous approach) pinned the top hit of each to 1.0 regardless of how
    # weak it was, which made an unrelated image outrank a real document.
    #
    # Instead each modality is mapped onto a shared 0-1 scale through its own
    # measured noise floor and strong-match ceiling. Values below the floor are
    # noise: image hits under it are dropped so unrelated pictures stop
    # appearing, text hits are clamped to 0 so document recall is unchanged.
    search_text_score_floor: float = 0.40
    search_text_score_ceil: float = 0.80
    search_image_score_floor: float = 0.18
    search_image_score_ceil: float = 0.35


    # Code needs its own band even though it shares bge with documents.
    #
    # Two things push code's noise floor well above a document's. Every chunk
    # is embedded with an English header naming its path and symbol, which
    # lifts its similarity to *any* English query; and a single source file
    # yields dozens of chunks, so it gets dozens of chances to surface. The
    # result was that scoring code through the document floor of 0.40 turned
    # bge's noise range into a confident-looking 48-55% match.
    #
    # Measured against an indexed 1100-line perception module: unrelated
    # queries ("a photo of mountains", "invoice amount due") peaked at 0.593,
    # while genuine ones ("kalman filter smoothing", "detect white lane
    # pixels") ran 0.719-0.881. The gap between those is empty, so the floor
    # sits inside it.
    #
    # Code is DROPPED below its floor rather than clamped, like images and
    # unlike documents: an unrelated function in a result list is a wrong
    # answer, not a weak one.
    search_code_score_floor: float = 0.63
    search_code_score_ceil: float = 0.85

    # LLM / RAG (Milestone 2)
    # Empty api key is allowed so the app still boots and search keeps working
    # without a key — only the /ask endpoint degrades, and it says so clearly.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 1024

    # How many document chunks get stuffed into the prompt as context. Kept
    # below search_top_k because context quality beats context quantity —
    # low-scoring chunks mostly add noise the model has to argue past.
    rag_top_k: int = 5
    rag_min_similarity: float = 0.15

    # How many images the Ask page shows beside an answer. Small on purpose:
    # these are visual context the model never reads, so a long grid of weak
    # matches would only compete with the answer for attention. Scored with
    # the search_image_score_* calibration, not rag_min_similarity, which is
    # on the raw bge scale and meaningless for CLIP.
    rag_image_top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
