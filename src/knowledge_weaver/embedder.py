"""OpenAI-compatible embedding client for Knowledge Weaver.

Supports remote mode only (POST to EMBEDDING_BASE_URL/embeddings).
Uses httpx for HTTP — no heavy ML libraries required.

Graceful degradation: if env vars are not set, get_embedder() returns None,
allowing the rest of the system to function without embeddings.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

# BGE-M3 native dimension
DEFAULT_DIMENSION = 1024


class EmbeddingClient:
    """OpenAI-compatible embedding client.

    Calls POST {base_url}/embeddings with Authorization: Bearer {api_key}
    and body {model, input: [...]}.
    """

    MAX_BATCH_SIZE = 10

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimension: int = DEFAULT_DIMENSION,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self._client = httpx.Client(timeout=timeout)
        self._is_local = "127.0.0.1" in base_url or "localhost" in base_url

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Returns the embedding vector, or an empty list on error.
        """
        results = self.embed_batch([text])
        return results[0] if results else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, auto-splitting into API-compliant batch sizes."""
        if not texts:
            return []

        all_results: list[list[float]] = []
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            chunk = texts[i:i + self.MAX_BATCH_SIZE]
            result = self._embed_chunk(chunk)
            if not result:
                all_results.extend([[] for _ in chunk])
            else:
                all_results.extend(result)
            # Rate-limit only for remote APIs
            if not self._is_local and i + self.MAX_BATCH_SIZE < len(texts):
                import time
                time.sleep(3.0)
        return all_results

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        import time
        url = f"{self.base_url}/embeddings"
        for attempt in range(3):
            try:
                resp = self._client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.model, "input": texts},
                )
                if resp.status_code == 429 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
                return [item["embedding"] for item in items]
            except httpx.HTTPStatusError as e:
                logger.warning("Embedding API HTTP %s: %s", e.response.status_code, e.response.text[:150])
                return []
            except httpx.RequestError as e:
                logger.warning("Embedding API request failed: %s", e)
                return []
        return []

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def get_embedder() -> EmbeddingClient | None:
    """Factory: create EmbeddingClient from env vars, or None if not configured.

    Reads: EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL
    Optional: EMBEDDING_DIMENSION (default: 768)

    Returns None if any of the three required vars is missing — no crash.
    """
    base_url = os.environ.get("EMBEDDING_BASE_URL", "").strip()
    api_key = os.environ.get("EMBEDDING_API_KEY", "").strip()
    model = os.environ.get("EMBEDDING_MODEL", "").strip()

    is_local = base_url and ("127.0.0.1" in base_url or "localhost" in base_url)

    if not base_url or not model:
        logger.debug(
            "Embedding not configured (need EMBEDDING_BASE_URL, "
            "EMBEDDING_MODEL). Embeddings disabled."
        )
        return None

    if not is_local and not api_key:
        logger.debug(
            "Remote embedding requires EMBEDDING_API_KEY. Embeddings disabled."
        )
        return None

    dimension_str = os.environ.get("EMBEDDING_DIMENSION", "").strip()
    dimension = int(dimension_str) if dimension_str else DEFAULT_DIMENSION

    return EmbeddingClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        dimension=dimension,
    )
