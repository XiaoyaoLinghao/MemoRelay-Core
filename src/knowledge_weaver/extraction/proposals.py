"""Validate untrusted LLM extraction proposals against stored sanitized evidence.

The model only proposes; the backend computes identity, source authority,
UTF-8 byte spans and trust grades (engineering §6, LLM-01):

* every evidence ref must reference a part of the sanitized input group; a
  support quote must be an exact substring of that stored part (secret check
  already happened before storage);
* authority/evidence grade is derived from the native role/origin, never from
  model output; assistant inference cannot become user_confirmed;
* project hints must resolve to a registered event project; unresolved
  proposals are kept as warnings and not guessed;
* proposed display text/value is re-checked with the shared secret policy and
  quarantined if it still contains secrets;
* explicit corrections require native user evidence (CLAIM-03);
* a task completion proposal may only reach ``verified`` when every referenced
  ``support_quote`` covers the complete stored sanitized part record and every
  acceptance condition and "satisfied" entry is explicitly recorded as
  successfully reviewed in that backend-recorded span (a bare ``exit_code=0``,
  a mere name mention, a checklist entry, or a FAILED/NOT RUN/pending result
  proves neither the command nor the task).

F04 round-5: a success word only grounds a condition when it is structurally
bound to that condition as its own predicate/conclusion (``security_review
passed`` / ``security_review 已通过`` / ``security_review: PASS`` /
``security_review has passed``). Only neutral function words may stand between
the condition name and the predicate; an intervening action/object (``security
review log upload succeeded``) or a negated/future/quantified/non-predicate form
(``not yet passed``, ``passed = false``, ``has zero passed checks``, ``是否通过``)
never binds. Independently, a referenced span that records *any* failure,
NOT RUN/pending, negation, future intent, false assignment, quantified result or
unknown/conditional wording vetoes the whole completion, even when a success
word also appears in the same span (one pass + one FAILED is an unresolved
conflict, not proof).

F04 round-6: the binding must cover the *complete conclusion*, not just the
name-to-success-word prefix. A success predicate grounds a condition only when
the rest of its clause is part of the result (neutral completion fillers, a full
count such as ``3 of 3 checks``): a question (``passed?`` / ``通过？``), a trailing
hypothesis/scoping modifier (``passed is hypothetical`` / ``passed per a log
upload`` / ``通过了吗`` / ``通过是假设``), or a false assignment (``passed = 0``) is
not a completed asserted result and fails closed, as does a same-sentence
qualifier clause (``passed, hypothetically``). This is an allow list on the
conclusion structure rather than an extended negative word list; unknown
wording stays unverified. The direction is deliberately fail closed: unknown
wording yields ``reported_done`` plus an auditable downgrade reason instead of a
false ``verified_done``; conservative false negatives are accepted, false
positives are not.

F04 round-7: the trusted fact is tightened from "a success word appears inside
the referenced span" to "the referenced span itself is a complete review-result
record". The *whole* backend-recorded ``support_quote`` span must be consumable
structurally: every sentence (and every comma-delimited clause inside it) must
be a complete success conclusion of one of the declared acceptance conditions
(optionally led by ``and``/``then``/``plus``), and a result count is only a full
result when ``n == d > 0``. Anything the grammar cannot consume makes the whole
span unconsumable, so the task stays ``reported_done``: a prefix label
(``Example output only:`` / ``Fictional transcript:`` / ``示例输出：``), a
trailing scoping/fiction/unfinished declaration (``These statements are
fictional examples.`` / ``Both reviews remain unfinished.`` / ``实际两项均不及格。``),
an interrogative sentence (``Did ...`` / ``...?`` / ``...？``), an invalid count
(``3 of 2`` / ``2/3`` / ``0 of 0``), ordinary prose next to the conclusion, or
any other unknown wording. This is a structural completeness rule, not a longer
negative word list; the negative result markers remain only as a
defence-in-depth veto. Unknown material fails closed and conservative false
negatives are explicitly accepted.

F04 round-7 also tightens the evidence *scope* for ``task.status`` (the round-6
review found that a model could quote only a clean excerpt of a dirty record):
verification is only granted when **every** referenced ``support_quote`` covers
the complete stored sanitized part record (normalized equality with the part
text). A narrow excerpt -- even one that itself reads like a complete
conclusion, e.g. quoting ``security_review passed`` out of ``Example output
only: security_review passed`` -- cannot verify and downgrades the claim to
``reported_done`` with ``backend_downgraded="conditions_not_grounded_in_evidence"``
plus a ``scope incomplete`` warning, while the cited span (start/end byte, body
sha) and the tool observation stay recorded for audit. The veto scan still
covers every referenced span; only *granting* is limited to complete-part
references. Case/separator/whitespace differences are normalized away by
:func:`_grounding_text` when comparing quote and part text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from knowledge_weaver import contracts
from knowledge_weaver.store.claims import (
    AUTHORITY_RANK,
    ClaimError,
    record_claim_revision_in_transaction,
)

EXCLUDED_ORIGINS = frozenset({"memory_injection", "maintenance"})

#: Proposal strings that describe an already-verified completion. The backend
#: never takes these from model output for a self-report (LLM-01/AC-36).
VERIFIED_LEVEL_STATUSES = frozenset({"verified", "verified_done", "passed"})
SELF_REPORT_ASSERTIONS = frozenset({"completion_report", "inference"})


def _authority_rank(authority: str) -> int:
    return int(AUTHORITY_RANK.get(authority, 0))


def _proposal_status(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("status") or "").lower()
    if isinstance(value, str):
        return value.lower()
    return ""


def _grounding_text(text: object) -> str:
    """Normalize evidence/condition text for case/separator-insensitive matching."""
    lowered = str(text).casefold()
    for separator in ("_", "-", "/"):
        lowered = lowered.replace(separator, " ")
    return " ".join(lowered.split())


def _condition_entries(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return [str(raw)]


# ---------------------------------------------------------------------------
# Result-class vocabulary (F04 round-5).
#
# The positive side is a *binding grammar*, not a word list: a success
# predicate only grounds a condition when it is that condition's own
# predicate/conclusion. The negative side is a set of generic *veto classes*:
# any hit in a referenced ``support_quote`` span makes the completion
# unverifiable regardless of success words elsewhere (fail closed). Neither
# list is keyed to the independent reviewer's example sentences; they are
# general result classes, and the veto list can only ever withhold
# verification.
# ---------------------------------------------------------------------------

#: Failure / not-executed / still-pending / negated / future / false /
#: quantified / unknown-or-conditional / hypothesis-or-theoretical result
#: classes. A referenced span that contains any of these cannot prove an
#: acceptance result: the conflict, or the mere absence of an actual outcome,
#: stays unresolved and the task is at most reported. Words are matched as whole
#: ASCII words / CJK substrings.
NEGATIVE_RESULT_MARKERS = (
    # explicit failure and failed execution states
    "failed",
    "fail",
    "failure",
    "failing",
    "fails",
    "failures",
    "error",
    "errored",
    "errors",
    "aborted",
    "cancelled",
    "canceled",
    # not executed / not performed review states
    "not run",
    "not executed",
    "not performed",
    "not reviewed",
    "not verified",
    "not started",
    "skipped",
    "skips",
    "skip",
    # still pending / deferred states
    "pending",
    "todo",
    "tbd",
    "blocked",
    "deferred",
    "postponed",
    "unverified",
    "incomplete",
    # negation as a generic function-word class (veto only, never grants)
    "not",
    "no",
    "never",
    "nor",
    "neither",
    "without",
    "cannot",
    "can't",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "hasn't",
    "haven't",
    "isn't",
    "wasn't",
    "weren't",
    "aren't",
    "won't",
    "wouldn't",
    "shouldn't",
    # future intent / not yet an actual result
    "yet to",
    "scheduled",
    "planned",
    "tomorrow",
    "upcoming",
    "next week",
    "next month",
    "later",
    "soon",
    "will",
    "would",
    "shall",
    "should",
    "expected to",
    "to be verified",
    "to be reviewed",
    "to be executed",
    "to be run",
    "to be performed",
    "to be checked",
    "to be confirmed",
    "to be completed",
    "to be validated",
    # false assignment / quantified non-predicate usage
    "false",
    "untrue",
    "incorrect",
    "wrong",
    "zero",
    "none",
    "null",
    # unknown / conditional / hearsay wording (a hypothesis or a claim about a
    # result is not the recorded result itself)
    "whether",
    "if",
    "unless",
    "assuming",
    "suppose",
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "apparently",
    "reportedly",
    "allegedly",
    "supposedly",
    "presumably",
    "claimed",
    "claims",
    "unknown",
    "unclear",
    "unconfirmed",
    # hypothesis / theoretical qualification (a hypothesis about a result, or a
    # result that only holds in theory, is not the recorded result itself; the
    # class completion is not tied to one sentence position, so a qualifier in a
    # later sentence of the same span still vetoes)
    "hypothesis",
    "hypotheses",
    "hypothetical",
    "hypothetically",
    "hypothesize",
    "hypothesized",
    "assume",
    "assumes",
    "assumed",
    "assumption",
    "assumptions",
    "theoretical",
    "theoretically",
    "in theory",
    # failure colour in review dashboards
    "red",
    # Chinese: failure / not executed / pending
    "失败",
    "不正常",
    "不合格",
    "未成功",
    "不成功",
    "未执行",
    "未运行",
    "未进行",
    "未完成",
    "未通过",
    "未验证",
    "未复核",
    "未确认",
    "未开始",
    "未就绪",
    "未达标",
    "待执行",
    "待运行",
    "待验证",
    "待办",
    "待通过",
    "待复核",
    "待确认",
    "待完成",
    "待定",
    "待处理",
    "跳过",
    "尚无",
    "尚未",
    "中止",
    "取消",
    "错误",
    "异常",
    "无法验证",
    "不能验证",
    "未能验证",
    "无法通过",
    "不能通过",
    # Chinese: negation / future / quantified / non-predicate classes
    "并未",
    "并非",
    "不是",
    "未经",
    "从未",
    "还未",
    "没有",
    "没能",
    "是否",
    "与否",
    "能否",
    "据称",
    "据说",
    "声称",
    "据报道",
    "疑似",
    "据传",
    "传闻",
    "预计",
    "计划",
    "即将",
    "将要",
    "明天",
    "下月",
    "下个月",
    "稍后",
    "未来",
    "下次",
    "以后",
    "零",
    # Chinese: hypothesis / theoretical qualification (same class as the
    # English entries above; CJK markers match as substrings, so 假设 covers
    # 假设性/假设的 and 理论 covers 理论上/仅理论)
    "假设",
    "假定",
    "假想",
    "理论",
    "理论上",
    "仅理论",
    "推测",
    "猜测",
)

#: Success *predicates* that may serve as a condition's recorded conclusion.
#: Deliberately predicate/adjective forms only: a free-floating noun such as
#: ``success`` is not a conclusion and is absent, so ``security_review success``
#: stays unverified. A marker only counts when the binding grammar ties it to
#: the condition name through a neutral gap.
SUCCESS_PREDICATE_MARKERS = (
    "passed",
    "pass",
    "passes",
    "succeeded",
    "succeeds",
    "successful",
    "ok",
    "okay",
    "green",
    "verified",
    "completed",
    "done",
    "通过",
    "成功",
    "已完成",
    "已通过",
    "已验证",
    "合格",
    "正常",
    "完成",
    "无误",
)

#: Clause separators. Clause bounds are used by the binding grammar: the
#: condition's own predicate must follow the name inside the same clause.
_CLAUSE_SEPARATORS = ("\n", "\r", ";", "；", "。", "！", "？", "|", ",", "，", ".", "!", "?")

#: Neutral function words (and review-artifact nouns) that may stand between a
#: condition name and its own result predicate. This is the binding grammar's
#: "allowed gap": anything else is an action/object of its own, which separates
#: the success word from the condition and therefore cannot bind. Example:
#: ``security_review log upload succeeded`` has gap ``log upload`` (not neutral)
#: so ``succeeded`` is not the review's result.
PREDICATE_CONNECTORS = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "has",
        "have",
        "had",
        "having",
        "do",
        "does",
        "did",
        "now",
        "just",
        "finally",
        "successfully",
        "already",
        "also",
        "all",
        "check",
        "checks",
        "test",
        "tests",
        "verification",
        "result",
        "results",
        "status",
        "criteria",
    }
)

#: Chinese connector morphemes that may glue a condition name to its predicate
#: result (``security_review 已通过`` / ``security_review 均已通过``). A
#: non-connector morpheme such as ``待``/``并非`` breaks the binding.
ZH_PREDICATE_CONNECTORS = ("均已", "已经", "已", "均", "都", "也", "全部", "顺利", "是", "为")

#: Punctuation/decoration that is transparent inside the allowed gap.
_GAP_TRANSPARENT = frozenset(" \t:：=()（）[]【】{}「」<>《》\"'“”‘’*•·—–、")

_ZH_CONNECTORS_LONGEST_FIRST = tuple(sorted(ZH_PREDICATE_CONNECTORS, key=len, reverse=True))

#: Partial-result forms directly after a bound success predicate: "passed 1 of 3
#: checks", "passed 0/3" (the normaliser turns "/" into a space), "passed 2 4".
#: A partial count is not a full result: it is treated like a conflict and fails
#: closed instead of verifying. "passed 3 of 3" is a full result and still binds.
_PARTIAL_RESULT_PATTERNS = (
    re.compile(r"(?<![0-9a-z])(\d+)\s+(?:of|out\s+of)\s+(\d+)(?![0-9])"),
    re.compile(r"(?<![0-9a-z])(\d+)\s+(\d+)(?![0-9])"),
)

#: Sentence separators. A condition's *complete* conclusion may not be extended
#: by a continuation clause of the same sentence; only a genuine separate
#: sentence ends it. Commas are not sentence separators: a comma often attaches
#: a qualifier (``passed, hypothetically``) or coordinates two conclusions.
_SENTENCE_SEPARATORS = tuple(separator for separator in _CLAUSE_SEPARATORS if separator not in (",", "，"))

#: Neutral completion fillers and review-artifact nouns that may follow a bound
#: success predicate as part of the *result itself* (``passed successfully``,
#: ``passed all checks``, ``通过检查``). This is an allow list on the conclusion
#: structure, not another negative word list: any other material after the
#: predicate means the clause does not state a completed, concluded result (a
#: hypothesis suffix ``is hypothetical``, a scoping modifier ``per a log
#: upload``, a false assignment ``= 0``, an interrogative particle ``了吗``), so
#: it cannot ground verification. Unknown wording fails closed.
_RESULT_TAIL_WORDS = frozenset(
    {
        # neutral completion adverbs
        "successfully",
        "already",
        "now",
        "just",
        "finally",
        "also",
        "all",
        # review-artifact nouns that may qualify a full result/count
        "check",
        "checks",
        "test",
        "tests",
        "case",
        "cases",
        "item",
        "items",
        "criteria",
        "verification",
        "verifications",
        "result",
        "results",
        "status",
        "review",
        "reviews",
        "step",
        "steps",
        # Chinese completion particle / review-artifact nouns. CJK text is split
        # on whitespace only, so glued forms such as ``了吗``/``是假设`` stay one
        # token and never match this filler list.
        "了",
        "检查",
        "测试",
        "复核",
        "审核",
        "验收",
        "验证",
        "结果",
        "项",
        "用例",
    }
)

#: A completed full-result count directly after the predicate: ``3 of 3``,
#: ``3 of 3 checks``, ``2 2`` (normalised ``2/2``). Numerator/denominator are
#: validated separately: an empty set (``0 of 0``) or a partial count is not a
#: completed result.
_COMPLETE_COUNT_PREFIX = re.compile(r"^(\d+)(?:\s+(?:of|out\s+of))?\s+(\d+)(?![0-9])")

#: Coordination words that may lead a further clause of the same sentence while
#: still only coordinating another *complete* success conclusion
#: (``passed, and restore_validation passed``).
_CONTINUATION_LEAD_WORDS = frozenset({"and", "then", "plus"})


def _marker_present(text: str, marker: str) -> bool:
    """Match ASCII markers as whole words; Chinese markers as substrings."""
    if marker.isascii():
        return re.search(rf"(?<![0-9a-z]){re.escape(marker)}(?![0-9a-z])", text) is not None
    return marker in text


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    for separator in _CLAUSE_SEPARATORS:
        index = text.rfind(separator, 0, start)
        if index >= 0 and index + len(separator) > left:
            left = index + len(separator)
    right = len(text)
    for separator in _CLAUSE_SEPARATORS:
        index = text.find(separator, end)
        if index >= 0 and index < right:
            right = index
    return left, right


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Same as :func:`_clause_bounds` but bounded by sentence separators only."""
    left = 0
    for separator in _SENTENCE_SEPARATORS:
        index = text.rfind(separator, 0, start)
        if index >= 0 and index + len(separator) > left:
            left = index + len(separator)
    right = len(text)
    for separator in _SENTENCE_SEPARATORS:
        index = text.find(separator, end)
        if index >= 0 and index < right:
            right = index
    return left, right


def _marker_offsets(text: str, marker: str) -> list[int]:
    """Return every start offset of ``marker`` in ``text``.

    ASCII markers match as whole words (never inside ``passed``/``unpassed``),
    CJK markers match as substrings.
    """
    if marker.isascii():
        pattern = re.compile(rf"(?<![0-9a-z]){re.escape(marker)}(?![0-9a-z])")
        return [match.start() for match in pattern.finditer(text)]
    offsets: list[int] = []
    index = text.find(marker)
    while index >= 0:
        offsets.append(index)
        index = text.find(marker, index + 1)
    return offsets


def _neutral_gap(gap: str) -> bool:
    """True when ``gap`` only contains neutral connectors/punctuation.

    This is the allowed-gap rule that binds a success predicate to a condition
    name. Fail closed: an unknown word is *not* neutral, so an unrelated action
    or object between the name and the success word breaks the binding.
    """
    cleaned = "".join(character for character in gap if character not in _GAP_TRANSPARENT)
    for connector in _ZH_CONNECTORS_LONGEST_FIRST:
        cleaned = cleaned.replace(connector, " ")
    return all(token in PREDICATE_CONNECTORS for token in cleaned.split())


def _span_vetoes_verification(text: str) -> bool:
    """True when a referenced span records any negative/unresolved result class.

    Checked over the whole quoted span (not clause-by-clause and not only the
    first occurrence): a span that records a failure, NOT RUN/pending/future
    intent, negation, false assignment, quantified result or unknown/conditional
    wording cannot prove acceptance even when a success word also appears there.
    That unresolved positive/negative conflict is fail closed; the caller reports
    ungrounded conditions instead of granting ``verified``.
    """
    return any(_marker_present(text, marker) for marker in NEGATIVE_RESULT_MARKERS)


def _success_is_partial(suffix: str) -> bool:
    """True when the text after a bound success predicate records a partial count."""
    for pattern in _PARTIAL_RESULT_PATTERNS:
        for match in pattern.finditer(suffix):
            if int(match.group(1)) < int(match.group(2)):
                return True
    return False


def _result_tail_is_complete(suffix: str) -> bool:
    """True when ``suffix`` completes a bound success predicate into a full result.

    The conclusion of a review is not just the success word: the rest of its
    clause must belong to the *result*. Only neutral completion fillers
    (:data:`_RESULT_TAIL_WORDS`), a full review count such as ``3 of 3 checks``,
    and ordinary punctuation may follow the predicate. A count only completes a
    result when ``n == d > 0`` (round-7: ``3 of 2``, ``2/3`` and ``0 of 0`` are
    invalid/partial and fail closed). Anything else — a hypothesis suffix (``is
    hypothetical``), a scoping modifier (``per a log upload``), a false
    assignment (``= 0``), an interrogative particle (``吗``, ``了吗``) — means the
    clause does not state a completed, concluded result. This is an allow list on
    the conclusion structure (not another negative word list): unknown wording
    fails closed.
    """
    remainder = "".join(
        character if character not in _GAP_TRANSPARENT else " " for character in suffix
    ).strip()
    if not remainder:
        return True
    count = _COMPLETE_COUNT_PREFIX.match(remainder)
    if count is not None:
        numerator, denominator = int(count.group(1)), int(count.group(2))
        if numerator <= 0 or numerator != denominator:
            # Round-7: only a full, non-empty result count (n == d > 0) is a
            # completed review. A partial (2/3), over-counted (3 of 2) or empty
            # (0 of 0) count is not a result at all.
            return False
        remainder = remainder[count.end():].strip()
        if not remainder:
            return True
    return all(token in _RESULT_TAIL_WORDS for token in remainder.split())


def _fragment_asserts_success(fragment: str) -> bool:
    """True when ``fragment`` ends with a complete success conclusion.

    Used for a further clause of the same sentence that may only *coordinate*
    another concluded result (``passed, restore_validation passed``). The
    fragment must contain a success predicate that completes the fragment
    (neutral preface, neutral/full-count suffix); a qualifier clause such as
    ``hypothetically`` or ``which is hypothetical`` has no such predicate and is
    therefore not a complete conclusion.
    """
    for marker in SUCCESS_PREDICATE_MARKERS:
        for offset in _marker_offsets(fragment, marker):
            if not _neutral_gap(fragment[:offset]):
                continue
            suffix = fragment[offset + len(marker) :]
            if _success_is_partial(suffix):
                return False
            return _result_tail_is_complete(suffix)
    return False


def _sentence_continuation_is_complete(text: str, right: int, start: int, declared_conditions: list[str]) -> bool:
    """True when the rest of the condition's sentence only completes the result.

    A conclusion is complete only if the remainder of its *sentence* does not
    attach a qualifier to it (``passed, hypothetically`` / ``passed, which is
    hypothetical``). Commas are allowed to coordinate further conclusions for
    the proposed acceptance conditions (``passed, restore_validation passed``);
    every other comma clause must itself be a complete or neutral conclusion.
    Unknown continuation wording fails closed.
    """
    _, sentence_right = _sentence_bounds(text, start, right)
    continuation = text[right:sentence_right]
    if all(character in _GAP_TRANSPARENT or character.isspace() for character in continuation):
        return True
    declared = sorted((name for name in declared_conditions if name), key=len, reverse=True)
    for clause in continuation.replace("，", ",").split(","):
        normalized = clause
        for name in declared:
            normalized = normalized.replace(name, " ")
        tokens = normalized.split()
        while tokens and tokens[0] in _CONTINUATION_LEAD_WORDS:
            tokens.pop(0)
        normalized = " ".join(tokens)
        if not normalized:
            continue
        if not _fragment_asserts_success(normalized):
            return False
    return True


def _condition_predicate_result(text: str, start: int, end: int, declared_conditions: list[str]) -> str:
    """Classify one condition occurrence by the result predicate bound to it.

    Returns ``"success"`` only when a success predicate directly follows the
    condition name through a neutral gap *and* the whole conclusion is complete:
    the rest of the predicate's clause is part of the result (``passed``,
    ``has passed``, ``: PASS``, ``已通过``, ``passed 3 of 3 checks``) and no
    qualifier is attached to it (``passed`` followed only by neutral fillers or
    a separate complete sentence). ``"partial"`` marks a partial count
    (``passed 1 of 3 checks``). ``"incomplete"`` marks a bound success word whose
    clause is not a completed conclusion: a question (``passed?`` / ``通过？``),
    a hypothesis/scoping suffix (``passed is hypothetical`` / ``passed per a log
    upload`` / ``通过了吗`` / ``通过是假设``), or a false assignment
    (``passed = 0``). ``""`` means no success predicate is bound at all (an
    unrelated action in the gap such as ``security_review log upload succeeded``,
    a pending morpheme such as ``security_review 待通过``, or unknown wording).
    """
    _, right = _clause_bounds(text, start, end)
    tail = text[end:right]
    interrogative = right < len(text) and text[right] in ("?", "？")
    for marker in SUCCESS_PREDICATE_MARKERS:
        for offset in _marker_offsets(tail, marker):
            if not _neutral_gap(tail[:offset]):
                continue
            suffix = tail[offset + len(marker) :]
            if _success_is_partial(suffix):
                return "partial"
            if interrogative or not _result_tail_is_complete(suffix):
                return "incomplete"
            if not _sentence_continuation_is_complete(text, right, start, declared_conditions):
                return "incomplete"
            return "success"
    return ""


#: Single-character sentence separators. A referenced span is parsed as a
#: sequence of sentences; every sentence (and every comma clause in it) must be a
#: complete success conclusion of a declared condition for the span to count as a
#: complete review-result record (F04 round-7).
_SENTENCE_SEPARATOR_CHARS = frozenset(_SENTENCE_SEPARATORS)
#: Comma-style clause separators (ASCII comma, full-width comma, CJK enumeration
#: comma). Every resulting clause must itself be a complete conclusion.
_COMMA_SEPARATOR_CHARS = frozenset((",", "，", "、"))

#: Characters that may decorate the edges of a clause without carrying meaning.
_EDGE_TRANSPARENT = "".join(sorted(_GAP_TRANSPARENT))


def _split_segments(text: str, separators: frozenset[str]) -> list[tuple[str, str]]:
    """Split ``text`` on single-character ``separators`` into (segment, terminator)."""
    segments: list[tuple[str, str]] = []
    current: list[str] = []
    for character in text:
        if character in separators:
            segments.append(("".join(current), character))
            current = []
            continue
        current.append(character)
    segments.append(("".join(current), ""))
    return segments


def _clause_conclusion(clause: str, declared_conditions: tuple[str, ...]) -> str | None:
    """Classify one comma-delimited clause of a referenced span (round-7 F04).

    Returns the normalized declared condition name when the clause *is* a
    complete success conclusion of that condition (``security_review passed`` /
    ``security_review: passed 3 of 3 checks`` / ``security_review 已通过``,
    optionally led by a coordination word ``and``/``then``/``plus``). Returns
    ``""`` for a clause that carries only decoration/whitespace (neutral), and
    ``None`` for any clause the completeness grammar cannot consume, which makes
    the whole referenced span fail closed.
    """
    stripped = clause.strip(_EDGE_TRANSPARENT)
    if not stripped:
        return ""
    tokens = stripped.split()
    while tokens and tokens[0] in _CONTINUATION_LEAD_WORDS:
        tokens.pop(0)
    if not tokens:
        return None
    candidate = " ".join(tokens)
    for name in declared_conditions:
        if not candidate.startswith(name):
            continue
        boundary = candidate[len(name) : len(name) + 1]
        if boundary and boundary.isascii() and (boundary.isalnum() or boundary == "_"):
            # The clause starts with a different word that merely shares the
            # declared name as a prefix; it is not that condition's conclusion.
            continue
        result = _condition_predicate_result(candidate, 0, len(name), list(declared_conditions))
        return name if result == "success" else None
    return None


def _span_complete_conclusions(
    text: str, declared_conditions: tuple[str, ...]
) -> set[str] | None:
    """Parse a whole referenced span as a complete review-result record (round-7 F04).

    The round-7 contract trusts a span only when the *entire* span is a review
    result record: every sentence must consist only of complete success
    conclusions for the declared acceptance conditions (comma-coordinated
    conclusions are allowed). A prefix label, a trailing scoping/fiction/
    unfinished declaration, an interrogative sentence or any other unconsumable
    material makes the span unrecorded (``None``), so the caller reports
    ``reported_done`` instead of verifying. Returns the set of declared
    conditions that are completed by a full conclusion in this span.
    """
    if not declared_conditions:
        return set()
    grounded: set[str] = set()
    for sentence, terminator in _split_segments(text, _SENTENCE_SEPARATOR_CHARS):
        if terminator in ("?", "？"):
            # A question-terminated sentence never grounds a completed result
            # (round-7 rule 5), even when the words before the terminal read like
            # a success conclusion.
            return None
        if not sentence.strip(_EDGE_TRANSPARENT):
            continue
        for clause, _separator in _split_segments(sentence, _COMMA_SEPARATOR_CHARS):
            conclusion = _clause_conclusion(clause, declared_conditions)
            if conclusion is None:
                return None
            if conclusion:
                grounded.add(conclusion)
    return grounded


def _condition_has_explicit_success(
    condition: str, evidence_spans: list[str], declared_conditions: list[str]
) -> bool:
    """Return True when the referenced spans record this condition's success.

    Only the backend-recorded support-quote spans are inspected, never the rest
    of the stored part (round-3 F04). Round-7 F04 changed the trusted fact from
    "a success word appears inside the span" to "the span itself is a complete
    review-result record". For every referenced span:

    * a span that records any negative/unresolved result class vetoes the
      completion outright (``_span_vetoes_verification``); one pass + one FAILED
      is an unresolved conflict, not proof;
    * the whole span must be structurally consumable as complete success
      conclusions of the declared conditions
      (``_span_complete_conclusions``). A prefix label (``Example output only:``
      / ``Fictional transcript:`` / ``示例输出：``), a trailing
      scoping/fiction/unfinished declaration (``Both reviews remain unfinished.``
      / ``实际两项均不及格。``), an interrogative sentence, ordinary prose or any
      unknown wording cannot be consumed and fails closed;
    * the condition must be completed by a full success conclusion in some span
      (``_clause_conclusion`` -> ``_condition_predicate_result``), and a result
      count is only a full result when ``n == d > 0`` (``3 of 3 checks``; not
      ``3 of 2`` / ``2/3`` / ``0 of 0``).

    Every qualifying span must be a complete record; grounding is never decided
    by the first occurrence or by a success word alone. Unknown wording fails
    closed (``reported_done`` plus warning, never ``verified``).
    """
    normalized_condition = _grounding_text(condition)
    if not normalized_condition:
        return False
    normalized_declared = tuple(
        sorted(
            {name for name in (_grounding_text(entry) for entry in declared_conditions) if name},
            key=len,
            reverse=True,
        )
    )
    grounded: set[str] = set()
    for span_text in evidence_spans:
        text = _grounding_text(span_text)
        if not text:
            continue
        if _span_vetoes_verification(text):
            return False
        conclusions = _span_complete_conclusions(text, normalized_declared)
        if conclusions is None:
            return False
        grounded |= conclusions
    return normalized_condition in grounded



def _ungrounded_conditions(value: dict, evidence_spans: list[str]) -> list[str]:
    """Return acceptance conditions/satisfied entries without explicit success proof.

    The backend-recorded text span of each referenced ``support_quote`` is the
    only accepted basis and the rest of the stored part is out of scope. A
    condition is grounded only when such a span states a *complete* concluded
    success for it: a success predicate bound to the condition name with no
    hypothesis/interrogative/false modifier attached. A bare name, a checklist
    entry, a FAILED/NOT RUN/pending mention, an unrelated success word, or any
    negated/future/false/quantified/unknown form never grounds a verified
    completion. Empty/whitespace conditions are ungrounded by definition (they
    can never prove a review step).
    """
    conditions = _condition_entries(value.get("conditions") or value.get("acceptance_conditions"))
    satisfied = _condition_entries(value.get("satisfied") or value.get("passed_conditions"))
    declared = [str(condition) for condition in conditions + satisfied]
    ungrounded: list[str] = []
    for condition in declared:
        if not _condition_has_explicit_success(condition, evidence_spans, declared):
            ungrounded.append(str(condition))
    return ungrounded


def normalize_task_proposal(
    *,
    value: object,
    facet: str,
    authority: str,
    assertion_kind: str,
    evidence_spans: list[str] | None = None,
    evidence_scope_complete: bool = False,
) -> tuple[object, str, list[str]]:
    """Backend-computed verification level for task claims (F04).

    A model proposal may describe a completion report, but it cannot grant
    ``verified`` status: that level comes from backend-derived source authority
    plus a non-self-report verification type. Assistant/tool text that is only
    ``reported`` is downgraded to ``reported_done`` and its self-declared
    conditions are dropped (they were not independently recorded). The same
    applies to an ``observed`` facet asserted by a self-report source.

    Even non-self-report tool/user evidence only proves what it actually says
    (TIME-02): each proposed acceptance condition and "satisfied" entry must be
    explicitly recorded as successfully reviewed in the backend-recorded span of
    a referenced ``support_quote``. Text outside those spans, a bare name
    mention, a checklist entry, an unrelated success word, and FAILED/NOT
    RUN/pending results do not count. The success word must be *bound* to the
    condition as its own predicate/conclusion (round-5 F04 binding grammar) and
    the conclusion must be complete (round-6 F04: the rest of the clause may not
    attach a question, hypothesis/scoping modifier or false assignment such as
    ``passed?`` / ``passed is hypothetical`` / ``passed per a log upload`` /
    ``passed = 0`` / ``通过了吗`` / ``通过是假设``); a referenced span recording any
    negative/future/false/quantified/unknown result class vetoes the completion
    because the conflict is unresolved. Round-7 F04 additionally requires the
    *whole* referenced span to be a complete review-result record: every
    sentence/comma clause must itself be a complete success conclusion of a
    declared condition, so a prefix label (``Example output only:`` /
    ``示例输出：``), a trailing scoping/fiction/unfinished declaration (``These
    statements are fictional examples.`` / ``Both reviews remain unfinished.`` /
    ``实际两项均不及格。``), an interrogative sentence (``Did ...`` / ``...?``) or
    an invalid count (``3 of 2`` / ``2/3`` / ``0 of 0``) leaves the whole span
    unconsumable. Round-7 F04 additionally requires the evidence *scope* to be
    complete: ``evidence_scope_complete`` must be true, i.e. every referenced
    ``support_quote`` must cover the complete stored sanitized part record (see
    :func:`resolve_proposal`). A narrow excerpt -- even a clean one -- cannot
    grant verification. Ungrounded conditions or an incomplete evidence scope
    downgrade the whole completion to ``reported_done`` with a
    ``backend_downgraded`` marker, so neither a bare ``exit_code=0``, a
    condition name appearing in negative, qualified, unrelated-success or
    example/未完成 text, nor a clean excerpt of a dirty record can verify an
    unrelated security/restore acceptance list. The direction is fail closed:
    unknown wording stays reported, never verified; default is fail closed, so a
    caller that cannot prove the scope must leave ``evidence_scope_complete``
    false.
    """
    warnings: list[str] = []
    resolved_facet = facet
    resolved_value = value
    if _authority_rank(authority) < _authority_rank("tool_verified"):
        if resolved_facet == "observed":
            resolved_facet = "reported"
            warnings.append("observed facet from self-reported evidence downgraded to reported")
        if _proposal_status(resolved_value) in VERIFIED_LEVEL_STATUSES:
            if isinstance(resolved_value, dict):
                resolved_value = {
                    **resolved_value,
                    "status": "reported_done",
                    "conditions": [],
                    "satisfied": [],
                    "backend_downgraded": "verified_requires_verified_evidence",
                }
            else:
                resolved_value = "reported_done"
            warnings.append("verified-level task status from self-reported evidence downgraded")
    elif assertion_kind in SELF_REPORT_ASSERTIONS:
        if _proposal_status(resolved_value) in VERIFIED_LEVEL_STATUSES:
            if isinstance(resolved_value, dict):
                resolved_value = {
                    **resolved_value,
                    "status": "reported_done",
                    "conditions": [],
                    "satisfied": [],
                    "backend_downgraded": "completion_report_cannot_verify",
                }
            else:
                resolved_value = "reported_done"
            warnings.append("completion report cannot grant verified status")
    else:
        if _proposal_status(resolved_value) in VERIFIED_LEVEL_STATUSES and isinstance(resolved_value, dict):
            ungrounded = _ungrounded_conditions(resolved_value, list(evidence_spans or []))
            if ungrounded or not evidence_scope_complete:
                resolved_value = {
                    **resolved_value,
                    "status": "reported_done",
                    "backend_downgraded": "conditions_not_grounded_in_evidence",
                }
                if not evidence_scope_complete:
                    warnings.append(
                        "verified-level task status downgraded: cited support_quote is not the "
                        "complete stored record (evidence scope incomplete: every referenced "
                        "support_quote must cover the whole sanitized part)"
                    )
                if ungrounded:
                    warnings.append(
                        "verified-level task status downgraded: acceptance conditions lack "
                        "grounding in the stored evidence (" + ", ".join(ungrounded) + ")"
                    )
    return resolved_value, resolved_facet, warnings


class ExtractionError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class SanitizedPart:
    event_id: str
    event_revision: int
    part_index: int
    text: str
    body_sha256: str
    role: str
    origin_kind: str
    actor_id: str
    event_time: dict
    project_ids: tuple[str, ...]
    workspace_id: str


@dataclass(frozen=True)
class ResolvedClaim:
    workspace_id: str
    project_id: str | None
    subject_id: str
    predicate: str
    facet: str
    qualifiers: dict
    value: object
    display_text: str
    authority: str
    assertion_kind: str
    event_time: dict
    evidence: tuple[dict, ...]


def load_event_group(store, event_id: str, revision: int) -> dict:
    """Load one complete sanitized revision group for extraction."""
    with store.read() as conn:
        event = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
        if event is None:
            raise ExtractionError("SOURCE_NOT_FOUND", "event not found", retryable=False)
        revision_row = conn.execute(
            "SELECT * FROM event_revisions WHERE event_id = ? AND revision = ?",
            (event_id, int(revision)),
        ).fetchone()
        if revision_row is None:
            raise ExtractionError("SOURCE_NOT_FOUND", "event revision not found", retryable=False)
        if revision_row["group_state"] != "complete":
            raise ExtractionError("PARENT_PENDING", "event revision group is incomplete", retryable=True)
        if revision_row["lifecycle"] != "active":
            raise ExtractionError("SOURCE_EXPIRED", "event revision is not active", retryable=False)
        parts = conn.execute(
            "SELECT * FROM event_parts WHERE event_id = ? AND revision = ? ORDER BY part_index",
            (event_id, int(revision)),
        ).fetchall()
    event_time = {
        "value": revision_row["event_time_value"],
        "precision": revision_row["event_time_precision"],
        "timezone": revision_row["event_time_timezone"],
    }
    group = {
        "event_id": event["event_id"],
        "event_revision": revision_row["revision"],
        "workspace_id": event["workspace_id"],
        "project_ids": tuple(
            __import__("json").loads(revision_row["project_ids_json"])
        ),
        "event_time": event_time,
        "origin_kind": event["origin_kind"],
        "lifecycle": revision_row["lifecycle"],
        "parts": [
            {
                "event_id": event["event_id"],
                "event_revision": revision_row["revision"],
                "part_index": part["part_index"],
                "text": part["body_text"],
                "body_sha256": part["body_sha256"],
                "role": event["role"],
                "origin_kind": event["origin_kind"],
                "actor_id": event["actor_id"],
                "event_time": event_time,
                "project_ids": tuple(
                    __import__("json").loads(revision_row["project_ids_json"])
                ),
                "workspace_id": event["workspace_id"],
                "availability": part["availability"],
            }
            for part in parts
        ],
        "redaction_status": revision_row["redaction_status"],
    }
    # A model cannot reproduce a stable backend identity from an unconstrained
    # natural-language name on every call. Offer scoped existing field anchors,
    # not previous records as new evidence. Scoped non-secret qualifiers are
    # essential to stable field identity; labels may contain ordinary fact values.
    # Legacy anchors containing secrets anywhere in their outbound fields are omitted.
    projects = list(group["project_ids"])
    project_filter = (
        "c.project_id IN (" + ",".join("?" for _ in projects) + ")"
        if projects else "c.project_id IS NULL"
    )
    from knowledge_weaver.retrieval.tokenizer import fts_query

    columns = (
        "SELECT c.claim_id, c.subject_id, c.project_id, c.predicate, c.facet, "
        "c.qualifiers_json, r.display_text, r.value_json, r.authority, cc.state, "
        "r.event_time_value, r.event_time_precision, r.event_time_timezone, r.created_at, "
        "r.revision_id, r.assertion_kind, r.correction_of, r.supersedes_json "
    )
    joins = (
        "JOIN current_claims cc ON cc.claim_id = c.claim_id "
        "JOIN claim_revisions r ON r.revision_id = "
        "json_extract(cc.selected_revision_ids_json, '$[0]') "
    )
    scoped = (
        "c.workspace_id = ? AND " + project_filter
        + " AND cc.state IN ('current', 'conflict', 'uncertain') "
    )
    args = (group["workspace_id"], *projects)
    expression = fts_query(" ".join(part["text"] or "" for part in group["parts"])[:4000])
    with store.read() as conn:
        rows = []
        if expression:
            rows = conn.execute(
                columns + "FROM claims c " + joins
                + "JOIN claim_fts ON claim_fts.revision_id = r.revision_id "
                + "WHERE " + scoped + " AND claim_fts MATCH ? "
                + "ORDER BY bm25(claim_fts), c.claim_id LIMIT 96",
                (*args, expression),
            ).fetchall()
        # Keyword matches come first so an old field does not disappear behind
        # newer unrelated facts. The bounded recent fallback helps short inputs.
        if len(rows) < 48:
            rows += conn.execute(
                columns + "FROM claims c " + joins + "WHERE " + scoped
                + "ORDER BY r.created_at DESC, c.claim_id LIMIT 96",
                args,
            ).fetchall()
    candidates = []
    seen = set()
    for item in rows:
        if item["claim_id"] in seen:
            continue
        seen.add(item["claim_id"])
        label = str(item["display_text"] or "")
        outgoing = {key: item[key] for key in ("claim_id", "project_id", "predicate", "facet")}
        outgoing["label"] = label
        outgoing["qualifiers"] = __import__("json").loads(item["qualifiers_json"])
        if contracts.secret_kinds(contracts.compact_json(outgoing)):
            continue
        from knowledge_weaver.store.claims import _ordering_time
        time_record = {key: item[key] for key in ("event_time_value", "event_time_precision", "event_time_timezone", "created_at", "assertion_kind", "correction_of")}
        time_record["supersedes"] = __import__("json").loads(item["supersedes_json"])
        time_record["links"] = []
        if item["assertion_kind"] == "explicit_correction":
            with store.read() as conn:
                time_record["links"] = [dict(row) for row in conn.execute(
                    "SELECT er.event_time_value AS source_event_time_value, er.event_time_precision AS source_event_time_precision "
                    "FROM evidence_links l JOIN event_revisions er ON er.event_id=l.event_id AND er.revision=l.event_revision WHERE l.revision_id=?", (item["revision_id"],))]
        candidates.append({
            "claim_id": item["claim_id"],
            "subject_id": item["subject_id"],
            "project_id": item["project_id"],
            "predicate": item["predicate"],
            "facet": item["facet"],
            "value": __import__("json").loads(item["value_json"]),
            "authority": item["authority"],
            "state": item["state"],
            "ordering_time": _ordering_time(time_record)[0],
            "label": label[:240],
            # Local-only: target validation still binds qualified fields.
            "qualifiers": __import__("json").loads(item["qualifiers_json"]),
        })
    group["claim_candidates"] = candidates[:48]
    return group


def sanitized_parts(group: dict) -> list[SanitizedPart]:
    visible = [part for part in group["parts"] if part["availability"] == "full" and part["text"] is not None]
    return [SanitizedPart(**{key: value for key, value in part.items() if key != "availability"}) for part in visible]


def _normalized_hint(hint: str) -> str:
    return " ".join(str(hint).strip().lower().split())


def subject_id_for(project_id: str | None, subject_hint: str) -> str:
    digest = contracts.sha256_hex(
        contracts.compact_json([project_id or "", _normalized_hint(subject_hint)])
    )
    return "subject:" + digest[:32]


def derive_authority(assertion_kind: str, parts: list[SanitizedPart]) -> str:
    if assertion_kind == "inference":
        return "inferred"
    roles = {part.role for part in parts}
    origins = {part.origin_kind for part in parts}
    if "derived_import" in origins:
        return "legacy_unknown"
    if roles == {"user"} and origins <= {"user_input"}:
        return "user_confirmed"
    if "tool" in roles and "tool_execution" in origins:
        return "tool_verified"
    return "reported"


def resolve_proposal(
    group: dict, proposal: dict, parts: list[SanitizedPart]
) -> tuple[list[ResolvedClaim], list[str]]:
    """Resolve one schema-valid proposal into backend-verified claim revisions.

    Round-7 F04 scope rule: for ``task.status`` claims, *granting* verification
    additionally requires every referenced ``support_quote`` to cover the complete
    stored sanitized part record (normalized equality with the part text). A
    narrow excerpt is still recorded as evidence (start/end byte, body sha) and
    still participates in the veto scan, but it cannot grant ``verified``: the
    claim is downgraded to ``reported_done`` with
    ``backend_downgraded="conditions_not_grounded_in_evidence"`` and a
    ``scope incomplete`` warning.
    """
    from knowledge_weaver.store import claims as claim_store

    resolved: list[ResolvedClaim] = []
    warnings: list[str] = []
    part_map = {(part.event_id, part.event_revision, part.part_index): part for part in parts}
    allowed_projects = set(group["project_ids"])
    for index, claim in enumerate(proposal.get("claims") or []):
        label = f"claim[{index}]"
        evidence: list[dict] = []
        evidence_parts: list[SanitizedPart] = []
        evidence_spans: list[str] = []
        evidence_scope_complete = True
        bad_evidence = None
        for reference in claim.get("evidence") or []:
            key = (
                reference.get("event_id"),
                int(reference.get("event_revision") or 0),
                int(reference.get("part_index") or 0),
            )
            part = part_map.get(key)
            if part is None:
                bad_evidence = "evidence is outside the sanitized input group"
                break
            quote = reference.get("support_quote")
            span = contracts.utf8_span(part.text, quote)
            if span is None:
                bad_evidence = "support quote is not an exact substring of the stored sanitized part"
                break
            if _grounding_text(quote) != _grounding_text(part.text):
                # Round-7 F04: a cited span that is only an excerpt of the stored
                # part cannot be trusted as the complete review record (the model
                # could quote a clean fragment of an example/fictional/unfinished
                # record). The exact span is still stored for audit below.
                evidence_scope_complete = False
            evidence.append(
                {
                    "event_id": part.event_id,
                    "event_revision": part.event_revision,
                    "part_index": part.part_index,
                    "start_byte": span[0],
                    "end_byte": span[1],
                    "body_sha256": part.body_sha256,
                    "availability": "full",
                }
            )
            evidence_parts.append(part)
            # Grounding uses only the backend-recorded quoted span of this part,
            # never the rest of the part text (round-3 F04: no borrowing from
            # text outside the reference).
            evidence_spans.append(part.text.encode("utf-8")[span[0] : span[1]].decode("utf-8"))
        if bad_evidence is not None:
            warnings.append(f"{label}: dropped ({bad_evidence})")
            continue
        if not evidence:
            warnings.append(f"{label}: dropped (no valid evidence)")
            continue
        if claim.get("assertion_kind") == "explicit_correction":
            if derive_authority(claim["assertion_kind"], evidence_parts) != "user_confirmed":
                warnings.append(f"{label}: dropped (explicit correction requires native user evidence)")
                continue
        project_hint = claim.get("project_hint")
        if project_hint is not None:
            if project_hint not in allowed_projects:
                warnings.append(f"{label}: unresolved project hint {project_hint!r} (not guessed)")
                continue
            project_id = project_hint
        elif len(allowed_projects) == 1:
            project_id = next(iter(allowed_projects))
        else:
            project_id = None
        display_text = claim["display_text"]
        value = claim["value"]
        value_text = contracts.compact_json(value) if not isinstance(value, str) else value
        if contracts.secret_kinds(display_text) or contracts.secret_kinds(value_text):
            warnings.append(f"{label}: quarantined (proposal still contains secret content)")
            continue
        authority = derive_authority(claim["assertion_kind"], evidence_parts)
        facet = claim["facet"]
        # completion_report is a contract task assertion regardless of a
        # provider's free-form property name. Route it through task downgrade
        # rules/views instead of silently bypassing them as foo.status.
        if claim["assertion_kind"] == "completion_report" and claim["predicate"] != "task.status":
            original_predicate = claim["predicate"]
            claim = dict(claim, predicate="task.status")
            if claim.get("target_claim_hint") is None:
                claim["subject_hint"] = claim["subject_hint"] + " / " + original_predicate
        subject_id = subject_id_for(project_id, claim["subject_hint"])
        target = claim.get("target_claim_hint")
        if target is not None:
            candidate = next(
                (entry for entry in group.get("claim_candidates", []) if entry["claim_id"] == target),
                None,
            )
            if (
                candidate is None
                or candidate["project_id"] != project_id
                or candidate["predicate"] != claim["predicate"]
                or candidate["qualifiers"] != (claim.get("qualifiers") or {})
            ):
                warnings.append(f"{label}: dropped (target_claim_hint is not a compatible scoped field)")
                continue
            # Facets remain separate claim keys, but share the referenced entity
            # identity: an observed port can refer to the entity of a desired
            # port without overwriting the desired value.
            subject_id = candidate["subject_id"]
        from knowledge_weaver.store.claims import _time_evidence, AUTHORITY_RANK
        from knowledge_weaver.store.db import utc_now
        incoming_time, anomalous_time = _time_evidence({
            "event_time_value": group["event_time"]["value"],
            "event_time_precision": group["event_time"]["precision"],
            "created_at": utc_now(),
        })
        # Unknown/same-day/future anomalies already remain conflicts/uncertain
        # under the normal projector. Guard only a normal later value that
        # could otherwise silently supersede an existing confirmed field.
        native_update_intent = any(part.role == "user" and part.origin_kind == "user_input" and any(word in ref["support_quote"] for word in ("更正", "改为", "调整为", "更新为")) for part, ref in zip(evidence_parts, claim["evidence"]))
        if target is None and claim["assertion_kind"] != "explicit_correction" and not native_update_intent and incoming_time is not None and not anomalous_time:
            collision = next((entry for entry in group.get("claim_candidates", [])
                if entry["project_id"] == project_id and entry["subject_id"] == subject_id
                and entry["predicate"] == claim["predicate"] and entry["facet"] == facet
                and entry["qualifiers"] == (claim.get("qualifiers") or {})
                and entry["value"] != claim["value"] and entry["state"] == "current"
                and AUTHORITY_RANK.get(authority, 0) >= AUTHORITY_RANK.get(entry["authority"], 0)
                and entry["ordering_time"] is not None
                and incoming_time > entry["ordering_time"]), None)
            if collision is not None:
                # A declared new field cannot silently overwrite the same key.
                # Keep its native source/job for bounded retry, rather than
                # invent a property/qualifier or change the original clock.
                raise ExtractionError("MODEL_UNAVAILABLE",
                    "FIELD_IDENTITY_COLLISION: null target_claim_hint collides with " + collision["claim_id"]
                    + ". Your null target declared a NEW field. Preserve that semantic choice by using a distinct predicate or explicit qualifiers. Do not switch to an existing target merely to silence validation. Only an actual update of the same qualified property may use its target id. Preserve both fields and source meaning.", retryable=True)
        if claim["predicate"] == "task.status":
            # The model only proposes; the backend decides the verification level
            # from the derived evidence authority, the explicitly quoted evidence
            # spans and the complete-record scope of those references, never from
            # proposal strings or unquoted part text.
            value, facet, task_warnings = normalize_task_proposal(
                value=value,
                facet=facet,
                authority=authority,
                assertion_kind=claim["assertion_kind"],
                evidence_spans=evidence_spans,
                evidence_scope_complete=evidence_scope_complete,
            )
            warnings.extend(f"{label}: {warning}" for warning in task_warnings)
        resolved.append(
            ResolvedClaim(
                workspace_id=group["workspace_id"],
                project_id=project_id,
                subject_id=subject_id,
                predicate=claim["predicate"],
                facet=facet,
                qualifiers=claim.get("qualifiers") or {},
                value=value,
                display_text=display_text,
                authority=authority,
                assertion_kind=claim["assertion_kind"],
                event_time=group["event_time"],
                evidence=tuple(evidence),
            )
        )
    return resolved, warnings


def apply_resolved_claims(
    conn,
    resolved: list[ResolvedClaim],
    *,
    model_version: str,
    prompt_version: str,
    policy_version: str,
) -> list[dict]:
    written = []
    for claim in resolved:
        written.append(
            record_claim_revision_in_transaction(
                conn,
                workspace_id=claim.workspace_id,
                project_id=claim.project_id,
                subject_id=claim.subject_id,
                predicate=claim.predicate,
                facet=claim.facet,
                qualifiers=claim.qualifiers,
                value=claim.value,
                display_text=claim.display_text,
                authority=claim.authority,
                assertion_kind=claim.assertion_kind,
                event_time=claim.event_time,
                evidence=list(claim.evidence),
            )
        )
    return written
