"""Optional HTTP embedding provider (stdlib only; local BGE server or external API).

The KW core does not bundle a model runtime. This provider speaks two common
shapes over HTTP:

* OpenAI-compatible ``POST {"model","input":[...]}`` -> ``{"data":[{"embedding":[...]}]}``;
* sentence-transformers style ``{"embeddings":[[...]]}``.

Authentication uses an **environment variable name** (``api_key_env``); the key
value is read at call time and never placed in CLI args, logs or error messages.
Vectors are L2-normalized client-side when ``normalization == "l2"`` so a mixed
server normalization cannot corrupt cosine ranking. Any transport/shape/dimension
failure raises :class:`EmbeddingUnavailable`; callers degrade to keyword recall
and keep the vector job pending/failed visibly.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request


class EmbeddingUnavailable(RuntimeError):
    """An embedding endpoint could not provide a usable vector.

    ``retryable`` distinguishes a transient transport/outage failure (the worker
    records a bounded retry) from a shape/dimension contract mismatch (rebuild
    or fix the model configuration; retrying the same request cannot help).
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = "EMBEDDING_MISMATCH"
        self.message = message
        self.retryable = retryable


class HttpEmbeddingProvider:
    def __init__(
        self,
        url: str,
        *,
        model_id: str,
        dimension: int,
        model_revision: str = "1",
        api_key_env: str | None = None,
        timeout: float = 30.0,
        normalization: str = "l2",
    ) -> None:
        if not url:
            raise ValueError("embedding url is required")
        self.url = url
        self.model_id = model_id
        self.dimension = int(dimension)
        self.model_revision = model_revision
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self.normalization = normalization

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if value:
                headers["Authorization"] = f"Bearer {value}"
        return headers

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return list(vector)
        return [value / norm for value in vector]

    def _parse(self, payload: dict, expected: int) -> list[list[float]]:
        vectors = None
        if isinstance(payload.get("data"), list):
            vectors = [item.get("embedding") for item in payload["data"]]
        elif isinstance(payload.get("embeddings"), list):
            vectors = payload["embeddings"]
        if not isinstance(vectors, list) or len(vectors) != expected:
            raise EmbeddingUnavailable(
                f"embedding response count mismatch (expected {expected})", retryable=False
            )
        parsed = []
        for vector in vectors:
            if not isinstance(vector, list) or not all(isinstance(value, (int, float)) for value in vector):
                raise EmbeddingUnavailable(
                    "embedding response contains a non-numeric vector", retryable=False
                )
            if len(vector) != self.dimension:
                raise EmbeddingUnavailable(
                    f"embedding dimension mismatch: expected {self.dimension}, got {len(vector)}",
                    retryable=False,
                )
            parsed.append(self._normalize([float(value) for value in vector]))
        return parsed

    def embed(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model_id, "input": list(texts)}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - configured endpoint
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Never include headers/body/key material in the error.
            raise EmbeddingUnavailable(f"embedding endpoint unreachable ({type(exc).__name__})") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise EmbeddingUnavailable(
                "embedding endpoint returned invalid JSON", retryable=True
            ) from None
        if not isinstance(payload, dict):
            raise EmbeddingUnavailable(
                "embedding endpoint returned a non-object response", retryable=False
            )
        return self._parse(payload, len(texts))
