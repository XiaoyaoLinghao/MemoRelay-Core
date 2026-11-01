"""KW 2.0 optional embedding providers (offline hash for smoke tests, HTTP for BGE/API)."""

from .http_provider import EmbeddingUnavailable, HttpEmbeddingProvider
from .index import (
    configure_model,
    embedding_text,
    load_vectors,
    run_embed_worker_once,
    vector_ranking,
)

__all__ = [
    "EmbeddingUnavailable",
    "HttpEmbeddingProvider",
    "configure_model",
    "embedding_text",
    "load_vectors",
    "run_embed_worker_once",
    "vector_ranking",
]
