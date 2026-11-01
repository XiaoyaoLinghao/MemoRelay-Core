"""Dependency-free Chinese/English tokenization for FTS5 keyword recall.

SQLite's ``unicode61`` tokenizer does not segment Chinese, so the KW core
indexes a deterministic CJK bigram expansion plus Latin/number words. Query and
index use the same function, which keeps matching symmetric without adding
jieba/pypinyin to the clean-install dependency set (AC-41). This is a
retrieval-quality tradeoff documented in the T08 review; alias/synonym
evaluation is a later tuning task.
"""

from __future__ import annotations

import re
import unicodedata

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_LATIN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:-]*")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def tokenize(text: str) -> list[str]:
    """Return unique CJK bigrams (single chars when isolated) and word tokens."""
    normalized = normalize_text(text)
    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)

    for match in _CJK_RE.finditer(normalized):
        run = match.group()
        if len(run) == 1:
            add(run)
        else:
            for index in range(len(run) - 1):
                add(run[index : index + 2])
    for match in _LATIN_RE.finditer(normalized):
        token = match.group().strip("._:-")
        if len(token) >= 2 or token.isdigit():
            add(token)
    return tokens


def fts_text(text: str) -> str:
    return " ".join(tokenize(text))


def fts_query(text: str, *, max_terms: int = 128) -> str | None:
    tokens = tokenize(text)[:max_terms]
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (CJK 1 char ~= 1 token, Latin ~4 chars)."""
    cjk = len(_CJK_CHAR_RE.findall(text or ""))
    rest = len(_CJK_CHAR_RE.sub("", text or ""))
    return max(1, cjk + (rest + 3) // 4)
