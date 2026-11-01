"""KW entry secret re-check (SEC-01).

Detection only: KW never rewrites an adapter body and never stores a body that
fails the shared `redact:1` policy. The policy data is frozen in
`contracts/v1/redaction-policy.json`, so the DMA redactor and this re-check use
the same rule set; the synthetic cases are validated by both repositories.
"""

from __future__ import annotations

from knowledge_weaver import contracts


def secret_kinds(text: str) -> list[str]:
    return contracts.secret_kinds(text)


def contains_secret(text: str) -> bool:
    return contracts.contains_secret(text)


def would_withhold(text: str) -> bool:
    return contracts.would_withhold(text)
