"""Shared OpenAI-compatible "classify each item" caller.

Three call sites (typed_relations.llm_type_pairs, tiebreak.llm_judge_pairs,
recover.llm_keep_filter) all chunk a list, prompt the model with numbered items,
and map verdicts back by index. This centralizes that plumbing so a fix lands
once, and adds the safety the hand-rolled copies lacked: the returned JSON must
have *exactly* the keys ``1..N`` for the chunk, else the whole chunk is treated
as failed — preventing a shifted/extra index from assigning one item's verdict
to a different item (which, for merges, would auto-merge the wrong pair).
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional


def parse_indexed_verdicts(content: str, n: int) -> Optional[dict]:
    """Parse model output into ``{"1":v, ..., "N":v}`` or return None.

    Returns None (caller skips the chunk) when the content is not JSON, not an
    object, or its keys do not COVER ``{"1".."N"}``. K8: require the expected
    indices to be a subset (every item judged) but tolerate harmless extra keys
    (e.g. a stray ``"note"``) instead of dropping the whole chunk — the caller only
    reads keys ``"1".."N"`` so extras are ignored, and missing indices still fail.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
    try:
        mapping = json.loads(cleaned)
    except Exception:
        return None
    if not isinstance(mapping, dict):
        return None
    norm = {str(k): v for k, v in mapping.items()}
    if not {str(j) for j in range(1, n + 1)}.issubset(norm.keys()):
        return None
    return norm


def classify_items(items: list, *, key: Callable, render: Callable,
                   system_prompt: str, user_prefix: str,
                   api_url: str, api_key: str, model: str,
                   chunk: int = 20, timeout: float = 90.0,
                   on_chunk_fail: Optional[Callable] = None) -> dict:
    """Map each item to a string verdict via a chunked chat model.

    ``key(item)`` -> result-dict key; ``render(item)`` -> the prompt line (without
    the ``[n]`` prefix). On a failed/invalid chunk, ``on_chunk_fail(batch)`` (if
    given) returns a ``{key: verdict}`` dict to merge in; default is to leave the
    chunk's items out of the result (undecided) — the safe default for both
    "judge" and "noise filter" uses.
    """
    import httpx

    url = api_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    out: dict = {}

    def _fail(batch):
        if on_chunk_fail is not None:
            out.update(on_chunk_fail(batch))

    for i in range(0, len(items), chunk):
        batch = items[i:i + chunk]
        lines = [f"[{j + 1}] {render(it)}" for j, it in enumerate(batch)]
        try:
            resp = httpx.post(
                url,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": model, "temperature": 0.1, "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prefix + "\n".join(lines)},
                ]},
                timeout=timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! chunk {i // chunk + 1} failed: {exc}")
            _fail(batch)
            continue
        mapping = parse_indexed_verdicts(content, len(batch))
        if mapping is None:
            print(f"  ! chunk {i // chunk + 1} unparseable / index mismatch; skipping")
            _fail(batch)
            continue
        for j, it in enumerate(batch):
            v = mapping.get(str(j + 1))
            if isinstance(v, str):
                out[key(it)] = v.strip()
    return out
