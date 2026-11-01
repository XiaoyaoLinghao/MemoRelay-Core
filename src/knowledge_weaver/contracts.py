"""Frozen DMA/KW shared-memory contract v1: loading, strict validation, semantic invariants.

This module is dependency-free on purpose: the KW 2.0 core must install on
macOS 13 / Intel / Python 3.11 without the MCP ``cryptography`` build chain
(see the baseline ``dependency-install.log`` evidence). It implements:

* contract file discovery (``contracts/v1`` in the repository or an installed
  package) and SHA-256 checksum verification;
* a strict JSON-Schema (2020-12 subset) validator covering exactly the keywords
  used by the frozen contracts, so unknown fields are always rejected;
* the pure ``event_id`` digest helper (ID-01), body digest and UTF-8 byte-span
  helpers, retract/chunk/group semantics (ID-02, RAW-01, SRC-03, SEC-02);
* batch-level shape, size, duplicate/conflict and group-consistency checks;
* the stable error catalog and the extraction proposal schema.

Nothing in this module performs I/O to the network or to production paths.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, NamedTuple

CONTRACT_VERSION = "1.0"
EVENT_SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = ("1.0",)
REDACTION_POLICY_VERSIONS = ("redact:1",)
EXTRACTION_SCHEMA_VERSION = "1.0"

MAX_PART_CHARS = 250_000
MAX_PART_BYTES = 1024 * 1024
MAX_BATCH_EVENTS = 100
MAX_BATCH_BYTES = 5 * 1024 * 1024

EVENT_ID_RE = re.compile(r"^evt:[a-f0-9]{64}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
EMPTY_TEXT = ""
EMPTY_BODY_SHA256 = hashlib.sha256(EMPTY_TEXT.encode("utf-8")).hexdigest()

# Engineering section 10 "minimum set" that the catalog must contain.
REQUIRED_ERROR_CODES = frozenset(
    {
        "REDACTION_REQUIRED",
        "UNSUPPORTED_SCHEMA",
        "INVALID_REQUEST",
        "UNREGISTERED_CLIENT",
        "SCOPE_DENIED",
        "PAYLOAD_TOO_LARGE",
        "EVENT_CONFLICT",
        "PARENT_PENDING",
        "SOURCE_EXPIRED",
        "SOURCE_NOT_FOUND",
        "MODEL_UNAVAILABLE",
        "EMBEDDING_MISMATCH",
        "STORE_UNAVAILABLE",
        "RESTORE_IN_PROGRESS",
        "STALE_CURSOR",
    }
)

REQUIRED_INVARIANT_IDS = frozenset(
    {
        "ID-01",
        "ID-02",
        "SRC-01",
        "SRC-02",
        "RAW-01",
        "SEC-01",
        "SEC-02",
        "SRC-03",
        "SRC-04",
        "CLAIM-03",
        "CLAIM-01",
        "CLAIM-02",
        "TIME-01",
        "TIME-02",
        "JOB-01",
        "VEC-01",
        "TTL-01",
        "TTL-02",
        "RESTORE-01",
        "RECALL-01",
        "EXPORT-01",
        "LLM-01",
    }
)

CONTRACT_FILES = (
    "openapi.json",
    "extraction-output.schema.json",
    "semantic-invariants.json",
    "error-catalog.json",
    "valid-examples.json",
    "invalid-examples.json",
    "redaction-policy.json",
    "redaction-examples.json",
    "VERSION.json",
)

#: Formats that the standard-library validator actually enforces. Contract
#: authors must not introduce other formats silently.
SUPPORTED_FORMATS = frozenset({"date", "date-time"})

_KNOWN_SCHEMA_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$anchor",
        "$ref",
        "$comment",
        "title",
        "description",
        "default",
        "deprecated",
        "readOnly",
        "writeOnly",
        "examples",
        "type",
        "enum",
        "const",
        "multipleOf",
        "maximum",
        "exclusiveMaximum",
        "minimum",
        "exclusiveMinimum",
        "maxLength",
        "minLength",
        "pattern",
        "maxItems",
        "minItems",
        "uniqueItems",
        "maxProperties",
        "minProperties",
        "required",
        "properties",
        "patternProperties",
        "additionalProperties",
        "propertyNames",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "items",
        "prefixItems",
        "contains",
        "maxContains",
        "minContains",
        "format",
        "contentMediaType",
        "contentEncoding",
    }
)

_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


class ContractError(RuntimeError):
    """Raised when the contract package cannot be located or is unusable."""


class Issue(NamedTuple):
    path: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.path}: {self.message}"


# ---------------------------------------------------------------------------
# Discovery / loading
# ---------------------------------------------------------------------------


def _candidate_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    candidates = [
        Path(os.environ["KW_CONTRACTS_DIR"])
        if os.environ.get("KW_CONTRACTS_DIR")
        else None,
        here.parents[2] / "contracts" / "v1" if len(here.parents) > 2 else None,
        Path.cwd() / "contracts" / "v1",
        here.parent / "contracts_v1",
    ]
    return [p for p in candidates if p is not None]


def contracts_dir() -> Path:
    """Return the frozen contract directory, honouring ``KW_CONTRACTS_DIR``."""
    tried = []
    for candidate in _candidate_dirs():
        tried.append(str(candidate))
        if (candidate / "VERSION.json").is_file() and (candidate / "openapi.json").is_file():
            return candidate
    raise ContractError(
        "frozen contract directory not found; tried: " + ", ".join(tried)
    )


def contract_path(name: str) -> Path:
    if name not in CONTRACT_FILES:
        raise ContractError(f"{name!r} is not a frozen contract file")
    path = contracts_dir() / name
    if not path.is_file():
        raise ContractError(f"missing frozen contract file: {path}")
    return path


def load_contract(name: str) -> Any:
    """Load one frozen contract file. Results are cached per process."""
    cached = _load_cache.get(name)
    if cached is not None:
        return cached
    try:
        value = json.loads(contract_path(name).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt checkout
        raise ContractError(f"contract file {name} is not valid JSON: {exc}") from exc
    _load_cache[name] = value
    return value


_load_cache: dict[str, Any] = {}


def openapi() -> dict:
    return load_contract("openapi.json")


def component_schemas() -> dict:
    return openapi().get("components", {}).get("schemas", {})


def extraction_schema() -> dict:
    return load_contract("extraction-output.schema.json")


def invariants() -> list[dict]:
    return load_contract("semantic-invariants.json")


def error_catalog() -> dict:
    return load_contract("error-catalog.json")


def valid_examples() -> dict:
    return load_contract("valid-examples.json")


def invalid_examples() -> list[dict]:
    return load_contract("invalid-examples.json")


def version_info() -> dict:
    return load_contract("VERSION.json")


def redaction_policy() -> dict:
    return load_contract("redaction-policy.json")


def redaction_examples() -> dict:
    return load_contract("redaction-examples.json")


def error_entry(code: str) -> dict | None:
    for entry in error_catalog().get("entries", []):
        if entry.get("code") == code:
            return entry
    return None


def is_retryable(code: str) -> bool:
    entry = error_entry(code)
    if entry is None:
        raise ContractError(f"unknown error code in contract: {code}")
    return bool(entry["retryable"])


def error_http_status(code: str) -> int:
    entry = error_entry(code)
    if entry is None:
        raise ContractError(f"unknown error code in contract: {code}")
    return int(entry["http_status"])


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_checksums() -> dict[str, str]:
    directory = contracts_dir()
    return {name: sha256_file(directory / name) for name in CONTRACT_FILES}


def verify_checksums() -> list[Issue]:
    """Verify ``SHA256SUMS`` covers exactly the frozen JSON files and matches."""
    directory = contracts_dir()
    issues: list[Issue] = []
    sums_path = directory / "SHA256SUMS"
    if not sums_path.is_file():
        return [Issue("SHA256SUMS", "missing checksum manifest")]
    declared: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 or not SHA256_RE.match(parts[0]):
            issues.append(Issue("SHA256SUMS", f"malformed entry: {line!r}"))
            continue
        declared[parts[1]] = parts[0]
    json_files = {p.name for p in directory.glob("*.json")}
    for name in sorted(json_files):
        if name not in declared:
            issues.append(Issue(name, "JSON file is not covered by SHA256SUMS"))
    for name, expected in sorted(declared.items()):
        path = directory / name
        if not path.is_file():
            issues.append(Issue(name, "listed in SHA256SUMS but missing"))
            continue
        actual = sha256_file(path)
        if actual != expected:
            issues.append(
                Issue(name, f"checksum mismatch: expected {expected}, got {actual}")
            )
    return issues


# ---------------------------------------------------------------------------
# Hashing helpers (ID-01 / body digest / UTF-8 spans)
# ---------------------------------------------------------------------------


def compact_json(value: Any) -> str:
    """Canonical compact JSON: UTF-8, unescaped Unicode, no extra whitespace."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_event_id(
    workspace_id: str, instance_id: str, native_event_id: str
) -> str:
    """ID-01: ``evt:`` + SHA-256 of the compact JSON array of the three strings."""
    payload = compact_json([workspace_id, instance_id, native_event_id])
    return "evt:" + sha256_hex(payload)


def body_digest(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def event_id_digest_of(event: dict) -> str:
    source = event.get("source") or {}
    return canonical_event_id(
        event.get("workspace_id", ""),
        source.get("instance_id", ""),
        source.get("native_event_id", ""),
    )


def utf8_span(text: str, quote: str) -> tuple[int, int] | None:
    """Return the UTF-8 byte span of the first exact occurrence of ``quote``."""
    if not isinstance(quote, str) or not quote:
        return None
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError:  # pragma: no cover - Python str is valid UTF-8
        return None
    needle = quote.encode("utf-8")
    start = encoded.find(needle)
    if start < 0:
        return None
    return start, start + len(needle)


def span_matches_hash(text: str, start: int, end: int, expected: str) -> bool:
    if start < 0 or end < start:
        return False
    encoded = text.encode("utf-8")
    if end > len(encoded):
        return False
    try:
        snippet = encoded[start:end].decode("utf-8")
    except UnicodeDecodeError:
        return False
    return body_digest(snippet) == expected or sha256_hex(encoded[start:end]) == expected


# ---------------------------------------------------------------------------
# Shared secret-detection engine (redact:1 policy)
# ---------------------------------------------------------------------------


def _compiled_policy() -> tuple[dict, list[tuple[str, "re.Pattern[str]"]], frozenset[str]]:
    policy = redaction_policy()
    rules = [(rule["kind"], re.compile(rule["pattern"])) for rule in policy["rules"]]
    return policy, rules, frozenset(policy.get("withhold_kinds") or [])


def redaction_policy_version() -> str:
    return str(redaction_policy()["policy_version"])


def secret_kinds(text: str) -> list[str]:
    """Return the policy rule kinds matched in ``text`` (never the values).

    This is detection only: KW uses it for the entry re-check (SEC-01) and never
    rewrites bodies; DMA uses the same policy data to redact before the outbox.
    """
    if not isinstance(text, str) or not text:
        return []
    _, rules, _ = _compiled_policy()
    kinds: list[str] = []
    for kind, pattern in rules:
        if pattern.search(text) and kind not in kinds:
            kinds.append(kind)
    return kinds


def contains_secret(text: str) -> bool:
    return bool(secret_kinds(text))


def would_withhold(text: str) -> bool:
    policy, _, withhold_kinds = _compiled_policy()
    return bool(set(secret_kinds(text)) & set(withhold_kinds))


def redact_text(text: str) -> tuple[str, list[str]]:
    """Return a policy-redacted copy of ``text`` plus the matched rule kinds.

    This is for **outbound** values only (e.g. a recall query that would
    otherwise be sent to an embedding/model endpoint). It never rewrites stored
    evidence bodies: the KW entry re-check stays detect-only for evidence, as
    decided in D006. Replacement values carry only the non-secret kind label.
    A matched ``withhold_kinds`` rule means the whole value must stay local; the
    caller has to drop the outbound copy instead of using the returned text.
    """
    if not isinstance(text, str) or not text:
        return "", []
    policy, rules, _withhold_kinds = _compiled_policy()
    template = str(policy.get("replacement_template") or "[REDACTED:{kind_upper}]")
    redacted = text
    kinds: list[str] = []
    for kind, pattern in rules:
        replacement = template.replace("{kind_upper}", kind.upper())
        new = pattern.sub(replacement, redacted)
        if new != redacted:
            kinds.append(kind)
            redacted = new
    return redacted, kinds


# ---------------------------------------------------------------------------
# JSON Schema (2020-12 subset) validator
# ---------------------------------------------------------------------------


def _type_ok(value: Any, type_name: str) -> bool:
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    expected = _JSON_TYPES.get(type_name)
    if expected is None:
        return False
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    return isinstance(value, expected)


def _format_ok(value: str, fmt: str) -> bool:
    if fmt == "date":
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            return False
        try:
            _dt.date.fromisoformat(value)
        except ValueError:
            return False
        return True
    if fmt == "date-time":
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            _dt.datetime.fromisoformat(text)
        except ValueError:
            return False
        return True
    # Unknown formats are annotations in 2020-12; contract health check rejects
    # them separately so they cannot silently go unenforced.
    return True


def _canonical_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ContractValidator:
    """Strict validator for the JSON-Schema subset used by contract v1."""

    def __init__(self, root_document: dict):
        self.root = root_document

    # -- reference resolution ------------------------------------------------
    def resolve_ref(self, ref: str) -> dict:
        if not ref.startswith("#"):
            raise ContractError(f"only local JSON references are supported: {ref}")
        node: Any = self.root
        pointer = ref[1:]
        if pointer in ("", "/"):
            return node
        for raw in pointer.lstrip("/").split("/"):
            token = raw.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict) and token in node:
                node = node[token]
            elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
                node = node[int(token)]
            else:
                raise ContractError(f"unresolvable JSON reference: {ref}")
        if not isinstance(node, dict):
            raise ContractError(f"JSON reference does not point at a schema: {ref}")
        return node

    # -- validation ----------------------------------------------------------
    def iter_issues(
        self,
        instance: Any,
        schema: dict,
        path: str = "$",
        ref_stack: tuple[str, ...] = (),
    ) -> Iterable[Issue]:
        if not isinstance(schema, dict):
            yield Issue(path, "schema node is not an object")
            return

        if "$ref" in schema:
            ref = schema["$ref"]
            if ref in ref_stack:
                yield Issue(path, f"cyclic JSON reference: {ref}")
                return
            try:
                target = self.resolve_ref(ref)
            except ContractError as exc:
                yield Issue(path, str(exc))
                return
            yield from self.iter_issues(
                instance, target, path, ref_stack + (ref,)
            )
            return

        if "const" in schema and not _json_equal(instance, schema["const"]):
            yield Issue(path, f"must equal constant {schema['const']!r}")
        if "enum" in schema:
            if not any(_json_equal(instance, allowed) for allowed in schema["enum"]):
                yield Issue(path, f"must be one of {schema['enum']!r}")

        type_spec = schema.get("type")
        types = type_spec if isinstance(type_spec, list) else ([type_spec] if type_spec else [])
        if types:
            if not any(_type_ok(instance, t) for t in types):
                yield Issue(path, f"expected type {types!r}, got {type(instance).__name__}")
                return

        if isinstance(instance, dict):
            yield from self._iter_object_issues(instance, schema, path, ref_stack)
        if isinstance(instance, list):
            yield from self._iter_array_issues(instance, schema, path, ref_stack)
        if isinstance(instance, str):
            yield from self._iter_string_issues(instance, schema, path)
        if isinstance(instance, (int, float)) and not isinstance(instance, bool):
            if "minimum" in schema and instance < schema["minimum"]:
                yield Issue(path, f"must be >= {schema['minimum']}")
            if "maximum" in schema and instance > schema["maximum"]:
                yield Issue(path, f"must be <= {schema['maximum']}")
            if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
                yield Issue(path, f"must be > {schema['exclusiveMinimum']}")
            if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
                yield Issue(path, f"must be < {schema['exclusiveMaximum']}")

        for keyword in ("allOf",):
            for index, subschema in enumerate(schema.get(keyword, [])):
                yield from self.iter_issues(
                    instance, subschema, path, ref_stack
                )
        if "anyOf" in schema:
            branches = schema["anyOf"]
            if not any(
                not list(self.iter_issues(instance, branch, path, ref_stack))
                for branch in branches
            ):
                yield Issue(path, "does not match anyOf branches")
        if "oneOf" in schema:
            matches = sum(
                1
                for branch in schema["oneOf"]
                if not list(self.iter_issues(instance, branch, path, ref_stack))
            )
            if matches != 1:
                yield Issue(path, f"matches {matches} oneOf branches, expected exactly 1")
        if "not" in schema:
            if not list(self.iter_issues(instance, schema["not"], path, ref_stack)):
                yield Issue(path, "must not match the 'not' schema")
        if "if" in schema:
            condition_ok = not list(self.iter_issues(instance, schema["if"], path, ref_stack))
            branch = schema.get("then") if condition_ok else schema.get("else")
            if branch is not None:
                yield from self.iter_issues(instance, branch, path, ref_stack)

    def _iter_object_issues(
        self, instance: dict, schema: dict, path: str, ref_stack: tuple[str, ...]
    ) -> Iterable[Issue]:
        for name in schema.get("required", []):
            if name not in instance:
                yield Issue(path, f"missing required property {name!r}")
        properties = schema.get("properties") or {}
        for name, value in instance.items():
            if name in properties:
                yield from self.iter_issues(
                    value, properties[name], f"{path}.{name}", ref_stack
                )
            elif "patternProperties" in schema and any(
                re.search(pattern, name)
                for pattern in schema["patternProperties"]
            ):
                for pattern, subschema in schema["patternProperties"].items():
                    if re.search(pattern, name):
                        yield from self.iter_issues(
                            value, subschema, f"{path}.{name}", ref_stack
                        )
            else:
                additional = schema.get("additionalProperties", True)
                if additional is False:
                    yield Issue(path, f"unknown property {name!r}")
                elif isinstance(additional, dict):
                    yield from self.iter_issues(
                        value, additional, f"{path}.{name}", ref_stack
                    )
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            yield Issue(path, f"must have at least {schema['minProperties']} properties")
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            yield Issue(path, f"must have at most {schema['maxProperties']} properties")

    def _iter_array_issues(
        self, instance: list, schema: dict, path: str, ref_stack: tuple[str, ...]
    ) -> Iterable[Issue]:
        if "minItems" in schema and len(instance) < schema["minItems"]:
            yield Issue(path, f"must have at least {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            yield Issue(path, f"must have at most {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            seen: set[str] = set()
            for index, item in enumerate(instance):
                key = _canonical_key(item)
                if key in seen:
                    yield Issue(f"{path}[{index}]", "duplicate array item")
                seen.add(key)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance):
                yield from self.iter_issues(
                    item, item_schema, f"{path}[{index}]", ref_stack
                )
        prefix = schema.get("prefixItems") or []
        for index, subschema in enumerate(prefix):
            if index < len(instance):
                yield from self.iter_issues(
                    instance[index], subschema, f"{path}[{index}]", ref_stack
                )

    def _iter_string_issues(
        self, instance: str, schema: dict, path: str
    ) -> Iterable[Issue]:
        if "minLength" in schema and len(instance) < schema["minLength"]:
            yield Issue(path, f"length must be >= {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            yield Issue(path, f"length must be <= {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            yield Issue(path, f"does not match pattern {schema['pattern']!r}")
        if "format" in schema and not _format_ok(instance, schema["format"]):
            yield Issue(path, f"is not a valid {schema['format']}")


def _json_equal(left: Any, right: Any) -> bool:
    return _canonical_key(left) == _canonical_key(right)


def validate_instance(schema_name: str, instance: Any) -> list[Issue]:
    """Validate one instance against a named ``components.schemas`` entry."""
    doc = openapi()
    schemas = doc.get("components", {}).get("schemas", {})
    if schema_name not in schemas:
        raise ContractError(f"unknown schema name: {schema_name}")
    wrapper = {"$ref": f"#/components/schemas/{schema_name}", **doc}
    return list(ContractValidator(wrapper).iter_issues(instance, wrapper))


def validate_extraction_output(instance: Any) -> list[Issue]:
    schema = extraction_schema()
    return list(ContractValidator(schema).iter_issues(instance, schema))


# ---------------------------------------------------------------------------
# Schema health (mirrors Draft202012Validator.check_schema for our subset)
# ---------------------------------------------------------------------------


def check_schema_structure() -> list[Issue]:
    """Check every component schema plus the extraction schema is well formed."""
    issues: list[Issue] = []
    doc = openapi()
    schemas = doc.get("components", {}).get("schemas", {})
    if not schemas:
        return [Issue("openapi", "no component schemas")]
    for name, schema in sorted(schemas.items()):
        issues.extend(_check_schema_node(doc, schema, f"schema:{name}", ref_ok=True))
    issues.extend(
        _check_schema_node(
            extraction_schema(), extraction_schema(), "schema:ExtractionOutput", ref_ok=False
        )
    )
    return issues


def check_openapi_surface() -> list[Issue]:
    """Check the public surface is complete and every reference resolves."""
    issues: list[Issue] = []
    doc = openapi()
    if doc.get("openapi") != "3.1.0":
        issues.append(Issue("openapi", "openapi version must be 3.1.0"))
    paths = doc.get("paths", {})
    expected = {
        "/v1/health",
        "/v1/events:batch",
        "/v1/recall",
        "/v1/corrections",
        "/v1/sources/{event_id}",
        "/v1/status",
        "/v1/capabilities",
        "/v1/resources",
    }
    missing = expected - set(paths)
    if missing:
        issues.append(Issue("openapi", f"missing paths: {sorted(missing)}"))
    validator = ContractValidator(doc)
    for path, operations in paths.items():
        for method, operation in operations.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue
            if "operationId" not in operation:
                issues.append(Issue(path, f"{method} missing operationId"))
            responses = operation.get("responses", {})
            if "200" not in responses:
                issues.append(Issue(path, f"{method} missing 200 response"))
            for status, response in responses.items():
                content = (response or {}).get("content", {})
                for media_type, body in content.items():
                    schema = body.get("schema", {})
                    try:
                        _walk_refs(validator, schema)
                    except ContractError as exc:
                        issues.append(Issue(f"{path}:{method}:{status}", str(exc)))
            if "parameters" in operation:
                for parameter in operation["parameters"]:
                    if "$ref" in parameter:
                        try:
                            validator.resolve_ref(parameter["$ref"])
                        except ContractError as exc:
                            issues.append(Issue(path, str(exc)))
            request_body = operation.get("requestBody")
            if request_body:
                for media_type, body in request_body.get("content", {}).items():
                    try:
                        _walk_refs(validator, body.get("schema", {}))
                    except ContractError as exc:
                        issues.append(Issue(f"{path}:{method}:requestBody", str(exc)))
    for name, schema in doc.get("components", {}).get("schemas", {}).items():
        if "schema_version" in schema.get("properties", {}):
            version_schema = schema["properties"]["schema_version"]
            if version_schema.get("const") != EVENT_SCHEMA_VERSION:
                issues.append(
                    Issue(
                        f"schema:{name}",
                        "schema_version must pin the frozen version "
                        f"{EVENT_SCHEMA_VERSION!r}",
                    )
                )
    return issues


def _walk_refs(validator: ContractValidator, node: Any) -> None:
    if isinstance(node, dict):
        if "$ref" in node:
            validator.resolve_ref(node["$ref"])
        for value in node.values():
            _walk_refs(validator, value)
    elif isinstance(node, list):
        for value in node:
            _walk_refs(validator, value)


def _check_schema_node(
    root: dict, node: Any, path: str, ref_ok: bool
) -> list[Issue]:
    issues: list[Issue] = []
    if not isinstance(node, dict):
        return [Issue(path, "schema node must be an object")]
    unknown = set(node) - _KNOWN_SCHEMA_KEYWORDS
    if unknown:
        issues.append(Issue(path, f"unknown JSON-Schema keywords: {sorted(unknown)}"))
    validator = ContractValidator(root)
    if "$ref" in node:
        if not isinstance(node["$ref"], str):
            issues.append(Issue(path, "$ref must be a string"))
        elif ref_ok:
            try:
                validator.resolve_ref(node["$ref"])
            except ContractError as exc:
                issues.append(Issue(path, str(exc)))
        return issues
    type_spec = node.get("type")
    type_names = type_spec if isinstance(type_spec, list) else [type_spec]
    if type_spec is not None:
        for type_name in type_names:
            if type_name not in _JSON_TYPES:
                issues.append(Issue(path, f"unknown type {type_name!r}"))
    if "format" in node and node["format"] not in SUPPORTED_FORMATS:
        issues.append(
            Issue(path, f"format {node['format']!r} is not enforced; extend validator first")
        )
    if "pattern" in node:
        try:
            re.compile(node["pattern"])
        except re.error as exc:
            issues.append(Issue(path, f"invalid pattern: {exc}"))
    if "const" in node and "type" in node:
        pass
    if "required" in node:
        required = node["required"]
        if not isinstance(required, list) or not all(isinstance(x, str) for x in required):
            issues.append(Issue(path, "'required' must be a list of strings"))
        elif type_spec == "object" and isinstance(node.get("properties"), dict):
            unknown_required = set(required) - set(node["properties"])
            if unknown_required:
                issues.append(
                    Issue(path, f"required properties not declared: {sorted(unknown_required)}")
                )
    if "enum" in node and not isinstance(node["enum"], list):
        issues.append(Issue(path, "'enum' must be a list"))
    for keyword in ("properties", "patternProperties", "dependentSchemas"):
        mapping = node.get(keyword)
        if mapping is not None and not isinstance(mapping, dict):
            issues.append(Issue(path, f"{keyword!r} must be an object"))
        elif isinstance(mapping, dict):
            for key, child in mapping.items():
                if keyword == "patternProperties":
                    try:
                        re.compile(key)
                    except re.error as exc:
                        issues.append(Issue(path, f"invalid patternProperty: {exc}"))
                issues.extend(_check_schema_node(root, child, f"{path}.{keyword}.{key}", ref_ok))
    if "additionalProperties" in node:
        extra = node["additionalProperties"]
        if not isinstance(extra, (bool, dict)):
            issues.append(Issue(path, "'additionalProperties' must be boolean or schema"))
        elif isinstance(extra, dict):
            issues.extend(_check_schema_node(root, extra, f"{path}.additionalProperties", ref_ok))
    if type_spec == "object" and isinstance(node.get("properties"), dict):
        if node.get("additionalProperties") is not False:
            issues.append(
                Issue(path, "strict contract requires additionalProperties: false")
            )
    if "items" in node:
        if not isinstance(node["items"], dict):
            issues.append(Issue(path, "'items' must be a schema"))
        else:
            issues.extend(_check_schema_node(root, node["items"], f"{path}.items", ref_ok))
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        branches = node.get(keyword)
        if branches is not None:
            if not isinstance(branches, list) or not branches:
                issues.append(Issue(path, f"{keyword!r} must be a non-empty list"))
            else:
                for index, child in enumerate(branches):
                    issues.extend(
                        _check_schema_node(root, child, f"{path}.{keyword}[{index}]", ref_ok)
                    )
    for keyword in ("not", "if", "then", "else", "propertyNames", "contains"):
        child = node.get(keyword)
        if child is not None:
            if not isinstance(child, dict):
                issues.append(Issue(path, f"{keyword!r} must be a schema"))
            else:
                issues.extend(
                    _check_schema_node(root, child, f"{path}.{keyword}", ref_ok)
                )
    for keyword in ("minLength", "maxLength", "minItems", "maxItems", "minimum", "maximum"):
        if keyword in node and not isinstance(node[keyword], (int, float)):
            issues.append(Issue(path, f"{keyword!r} must be a number"))
    return issues


# ---------------------------------------------------------------------------
# Semantic validation (beyond JSON Schema)
# ---------------------------------------------------------------------------


def validate_event_semantics(event: dict) -> list[Issue]:
    """Check ID-01 digests, part geometry, retract shape and revision arithmetic."""
    issues: list[Issue] = []
    schema_version = event.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        issues.append(
            Issue("$.schema_version", f"unsupported schema version {schema_version!r}")
        )
    source = event.get("source") or {}
    try:
        expected_id = event_id_digest_of(event)
    except (TypeError, ValueError):
        expected_id = None
    if expected_id is not None and event.get("event_id") != expected_id:
        issues.append(
            Issue("$.event_id", "ID-01: event_id must be the namespaced SHA-256 of "
                  "compact JSON [workspace_id, instance_id, native_event_id]")
        )
    body = event.get("body") or {}
    text = body.get("text")
    if isinstance(text, str):
        if len(text) > MAX_PART_CHARS:
            issues.append(Issue("$.body.text", f"part exceeds {MAX_PART_CHARS} characters"))
        byte_len = len(text.encode("utf-8"))
        if byte_len > MAX_PART_BYTES:
            issues.append(Issue("$.body.text", f"part exceeds {MAX_PART_BYTES} UTF-8 bytes"))
        expected_digest = body_digest(text)
        if body.get("sha256") != expected_digest:
            issues.append(
                Issue("$.body.sha256", "body digest must be SHA-256 of the redacted UTF-8 text")
            )
    part_index = body.get("part_index")
    part_count = body.get("part_count")
    if isinstance(part_index, int) and isinstance(part_count, int):
        if not 0 <= part_index < max(part_count, 1):
            issues.append(Issue("$.body.part_index", "part_index must be within [0, part_count)"))
        if part_count > MAX_BATCH_EVENTS:
            issues.append(Issue("$.body.part_count", "part_count exceeds batch part budget"))
    redaction = event.get("redaction") or {}
    policy = redaction.get("policy_version")
    if policy is not None and policy not in REDACTION_POLICY_VERSIONS:
        issues.append(
            Issue("$.redaction.policy_version", f"unsupported redaction policy {policy!r}")
        )
    if redaction.get("status") == "withheld" and body.get("text"):
        issues.append(
            Issue("$.body.text", "withheld content must not carry a body; store receipt only")
        )
    operation = event.get("operation")
    revision = event.get("event_revision")
    parent = event.get("parent_revision")
    if operation == "create":
        if revision != 1:
            issues.append(Issue("$.event_revision", "create must be revision 1"))
        if parent is not None:
            issues.append(Issue("$.parent_revision", "create cannot supersede another revision"))
    elif operation in ("edit", "retract", "redact"):
        if not isinstance(parent, int) or parent < 1:
            issues.append(Issue("$.parent_revision", f"{operation} requires a parent revision"))
        elif isinstance(revision, int) and revision != parent + 1:
            issues.append(
                Issue("$.event_revision", "revision must be parent_revision + 1")
            )
    if operation == "retract":
        if body.get("text") != EMPTY_TEXT:
            issues.append(Issue("$.body.text", "retract body must be one empty part"))
        if body.get("sha256") != EMPTY_BODY_SHA256:
            issues.append(Issue("$.body.sha256", "retract body must hash the empty string"))
        if body.get("part_index") != 0 or body.get("part_count") != 1:
            issues.append(Issue("$.body", "retract must be one empty part (part 0 of 1)"))
    return issues


_GROUP_METADATA_KEYS = (
    "workspace_id",
    "operation",
    "parent_revision",
    "event_time",
    "project_ids",
)


def validate_group_consistency(events: list[dict]) -> list[Issue]:
    """SRC-03 metadata agreement across parts of one revision (completeness excluded).

    Incomplete groups are legitimate: a revision may be split across several
    batches/requests, so only a complete group can be processed, but partial
    groups are not an error at batch-shape level.
    """
    issues: list[Issue] = []
    groups: dict[tuple[str, int], list[dict]] = {}
    for event in events:
        key = (event.get("event_id", ""), event.get("event_revision", 0))
        groups.setdefault(key, []).append(event)
    for (event_id, revision), parts in groups.items():
        label = f"{event_id}@{revision}"
        for field in _GROUP_METADATA_KEYS:
            values = {_canonical_key(part.get(field)) for part in parts}
            if len(values) > 1:
                issues.append(Issue(label, f"group metadata disagreement on {field!r}"))
        source_views = {
            _canonical_key(
                {
                    key: (part.get("source") or {}).get(key)
                    for key in (
                        "instance_id",
                        "host",
                        "actor_id",
                        "session_id",
                        "native_event_id",
                        "native_schema",
                        "role",
                        "origin_kind",
                        "turn_id",
                        "parent_event_id",
                    )
                }
            )
            for part in parts
        }
        if len(source_views) > 1:
            issues.append(Issue(label, "group metadata disagreement on source/role/origin"))
        content_revisions = {
            _canonical_key((part.get("body") or {}).get("content_revision"))
            for part in parts
        }
        if len(content_revisions) > 1:
            issues.append(Issue(label, "group metadata disagreement on content_revision"))
        redaction_views = {
            _canonical_key(
                {
                    key: (part.get("redaction") or {}).get(key)
                    for key in ("policy_version", "status")
                }
            )
            for part in parts
        }
        if len(redaction_views) > 1:
            issues.append(Issue(label, "group redaction status/policy disagreement"))
    return issues


def validate_event_group(events: list[dict]) -> list[Issue]:
    """SRC-03: metadata agreement plus completeness of every revision group."""
    issues = validate_group_consistency(events)
    groups: dict[tuple[str, int], list[dict]] = {}
    for event in events:
        key = (event.get("event_id", ""), event.get("event_revision", 0))
        groups.setdefault(key, []).append(event)
    for (event_id, revision), parts in groups.items():
        label = f"{event_id}@{revision}"
        part_count = (parts[0].get("body") or {}).get("part_count")
        if not isinstance(part_count, int) or part_count < 1:
            issues.append(Issue(label, "invalid part_count"))
            continue
        indexing = sorted(part.get("body", {}).get("part_index") for part in parts)
        if indexing != list(range(part_count)):
            issues.append(
                Issue(label, f"group incomplete or duplicated: parts {indexing}, count {part_count}")
            )
    return issues


def validate_batch_structure(batch: dict, encoded_bytes: int | None = None) -> list[Issue]:
    """Batch shape/size/identity checks for the KW entry point.

    These are whole-request errors (HTTP 400/413): schema, encoded size,
    duplicate part keys inside one batch and SRC-03 metadata disagreement.
    Per-part outcomes (duplicate/conflict/parent-pending/withheld/excluded) are
    decided by the store and returned as receipts.
    """
    issues: list[Issue] = validate_instance("Batch", batch)
    if issues:
        return issues
    events = batch.get("events") or []
    if len(events) > MAX_BATCH_EVENTS:
        issues.append(Issue("$.events", f"batch exceeds {MAX_BATCH_EVENTS} events"))
    if encoded_bytes is not None and encoded_bytes > MAX_BATCH_BYTES:
        issues.append(Issue("$", f"encoded batch exceeds {MAX_BATCH_BYTES} bytes"))
    seen: set[tuple[str, int, int]] = set()
    for index, event in enumerate(events):
        body = event.get("body") or {}
        key = (
            str(event.get("event_id")),
            int(event.get("event_revision") or 0),
            int(body.get("part_index") or 0),
        )
        if key in seen:
            issues.append(Issue(f"$.events[{index}]", "duplicate (event_id, revision, part_index) in batch"))
        seen.add(key)
    issues.extend(validate_group_consistency(events))
    return issues


def validate_batch_semantics(batch: dict, encoded_bytes: int | None = None) -> list[Issue]:
    """Batch shape + size + per-event + per-group semantics."""
    issues: list[Issue] = []
    events = batch.get("events")
    if not isinstance(events, list):
        return [Issue("$.events", "events must be an array")]
    if not events:
        issues.append(Issue("$.events", "batch must contain at least one event"))
    if len(events) > MAX_BATCH_EVENTS:
        issues.append(Issue("$.events", f"batch exceeds {MAX_BATCH_EVENTS} events"))
    if encoded_bytes is not None and encoded_bytes > MAX_BATCH_BYTES:
        issues.append(Issue("$", f"encoded batch exceeds {MAX_BATCH_BYTES} bytes"))
    seen: set[tuple[str, int, int]] = set()
    for index, event in enumerate(events):
        prefix = f"$.events[{index}]"
        for issue in validate_event_semantics(event):
            issues.append(Issue(f"{prefix}.{issue.path.lstrip('$.')}", issue.message))
        body = event.get("body") or {}
        key = (
            str(event.get("event_id")),
            int(event.get("event_revision") or 0),
            int(body.get("part_index") or 0),
        )
        if key in seen:
            issues.append(Issue(prefix, "duplicate (event_id, revision, part_index) in batch"))
        seen.add(key)
    issues.extend(validate_event_group(events))
    return issues


def validate_event(event: dict) -> list[Issue]:
    """Full validation for one event: schema then semantics."""
    return validate_instance("Event", event) + validate_event_semantics(event)


def validate_batch(batch: dict, encoded_bytes: int | None = None) -> list[Issue]:
    return validate_instance("Batch", batch) + validate_batch_semantics(batch, encoded_bytes)


# ---------------------------------------------------------------------------
# Catalog / invariant health
# ---------------------------------------------------------------------------


def check_error_catalog() -> list[Issue]:
    issues: list[Issue] = []
    catalog = error_catalog()
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        return [Issue("error-catalog", "missing entries")]
    codes: set[str] = set()
    for index, entry in enumerate(entries):
        path = f"entry[{index}]"
        code = entry.get("code")
        if not isinstance(code, str) or not TOKEN_RE.match(code):
            issues.append(Issue(path, "code must be an upper token"))
            continue
        if code in codes:
            issues.append(Issue(path, f"duplicate code {code}"))
        codes.add(code)
        if not isinstance(entry.get("http_status"), int) or not 400 <= entry["http_status"] <= 599:
            issues.append(Issue(path, "http_status must be 4xx/5xx"))
        if not isinstance(entry.get("retryable"), bool):
            issues.append(Issue(path, "retryable must be boolean"))
        for field in ("description", "client_action"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                issues.append(Issue(path, f"missing {field}"))
        if not isinstance(entry.get("scope"), list) or not set(entry["scope"]) <= {
            "http",
            "receipt",
        }:
            issues.append(Issue(path, "scope must be a subset of http/receipt"))
    missing = REQUIRED_ERROR_CODES - codes
    if missing:
        issues.append(Issue("error-catalog", f"missing required codes: {sorted(missing)}"))
    if catalog.get("schema_version") != EVENT_SCHEMA_VERSION:
        issues.append(Issue("error-catalog", "schema_version mismatch"))
    return issues


def check_invariants() -> list[Issue]:
    issues: list[Issue] = []
    seen: set[str] = set()
    for index, invariant in enumerate(invariants()):
        path = f"invariant[{index}]"
        invariant_id = invariant.get("id")
        rule = invariant.get("rule")
        if not isinstance(invariant_id, str) or not invariant_id:
            issues.append(Issue(path, "missing id"))
            continue
        if invariant_id in seen:
            issues.append(Issue(path, f"duplicate invariant id {invariant_id}"))
        seen.add(invariant_id)
        if not isinstance(rule, str) or len(rule) < 20:
            issues.append(Issue(path, f"{invariant_id}: missing substantive rule text"))
    missing = REQUIRED_INVARIANT_IDS - seen
    if missing:
        issues.append(Issue("semantic-invariants", f"missing invariants: {sorted(missing)}"))
    return issues


def version_consistency() -> list[Issue]:
    issues: list[Issue] = []
    info = version_info()
    checks = [
        ("contract_version", CONTRACT_VERSION),
        ("event_schema_version", EVENT_SCHEMA_VERSION),
        ("extraction_schema_version", EXTRACTION_SCHEMA_VERSION),
    ]
    for field, expected in checks:
        if info.get(field) != expected:
            issues.append(Issue("VERSION.json", f"{field} must be {expected!r}"))
    if info.get("supported_schema_versions") != list(SUPPORTED_SCHEMA_VERSIONS):
        issues.append(Issue("VERSION.json", "supported_schema_versions mismatch"))
    if info.get("checksums_file") != "SHA256SUMS":
        issues.append(Issue("VERSION.json", "checksums_file must be SHA256SUMS"))
    if info.get("redaction_policy_versions") != list(REDACTION_POLICY_VERSIONS):
        issues.append(Issue("VERSION.json", "redaction_policy_versions mismatch"))
    if redaction_policy_version() not in REDACTION_POLICY_VERSIONS:
        issues.append(Issue("redaction-policy.json", "policy_version not supported"))
    extraction = extraction_schema()
    if extraction.get("properties", {}).get("schema_version", {}).get("const") != (
        EXTRACTION_SCHEMA_VERSION
    ):
        issues.append(Issue("extraction-output.schema.json", "schema_version mismatch"))
    return issues


def shared_example_checks() -> tuple[list[Issue], list[dict]]:
    """Validate every shared example; return issues plus a per-check report."""
    issues: list[Issue] = []
    report: list[dict] = []
    api = openapi()
    schemas = api.get("components", {}).get("schemas", {})
    wrapper = ContractValidator({**api})
    for name in schemas:
        report.append({"check": f"schema:{name}", "passed": True})
    valid = valid_examples()
    for name, value in valid.items():
        found = validate_instance(name, value)
        report.append(
            {
                "check": f"valid:{name}",
                "passed": not found,
                "errors": [str(issue) for issue in found],
            }
        )
        issues.extend(found)
    for entry in invalid_examples():
        name = entry.get("schema")
        if name not in valid:
            issues.append(Issue("invalid-examples", f"no valid base example for {name!r}"))
            continue
        patched = json.loads(json.dumps(valid[name]))
        patched.update(entry.get("patch") or {})
        found = validate_instance(name, patched)
        passed = bool(found)
        report.append(
            {
                "check": f"invalid:{entry.get('reason')}",
                "passed": passed,
                "errors": [str(issue) for issue in found],
            }
        )
        if not passed:
            issues.append(
                Issue("invalid-examples", f"{entry.get('reason')!r} was accepted")
            )
    return issues, report


def check_redaction_policy() -> list[Issue]:
    """Validate the shared redact:1 policy data and its synthetic examples."""
    issues: list[Issue] = []
    try:
        policy = redaction_policy()
        examples = redaction_examples()
    except ContractError as exc:
        return [Issue("redaction-policy", str(exc))]
    version = policy.get("policy_version")
    if version not in REDACTION_POLICY_VERSIONS:
        issues.append(Issue("redaction-policy", f"unsupported policy_version {version!r}"))
    if examples.get("policy_version") != version:
        issues.append(Issue("redaction-examples", "policy_version does not match the policy"))
    if not isinstance(policy.get("rules"), list) or not policy["rules"]:
        issues.append(Issue("redaction-policy", "rules must be a non-empty list"))
        return issues
    kinds: set[str] = set()
    for index, rule in enumerate(policy["rules"]):
        path = f"rule[{index}]"
        kind = rule.get("kind")
        if not isinstance(kind, str) or not TOKEN_RE.match(kind):
            issues.append(Issue(path, "kind must be a short token"))
            continue
        if kind in kinds:
            issues.append(Issue(path, f"duplicate kind {kind}"))
        kinds.add(kind)
        try:
            re.compile(rule.get("pattern") or "")
        except (re.error, TypeError) as exc:
            issues.append(Issue(path, f"invalid pattern: {exc}"))
    withhold = set(policy.get("withhold_kinds") or [])
    if not withhold <= kinds:
        issues.append(Issue("redaction-policy", "withhold_kinds must reference declared rules"))
    template = policy.get("replacement_template") or ""
    if "{kind_upper}" not in template:
        issues.append(Issue("redaction-policy", "replacement_template must contain {kind_upper}"))
    seen_cases: set[str] = set()
    for index, case in enumerate(examples.get("cases") or []):
        path = f"case[{index}]"
        case_id = case.get("id")
        if not isinstance(case_id, str) or case_id in seen_cases:
            issues.append(Issue(path, "case id missing or duplicated"))
        seen_cases.add(str(case_id))
        raw = case.get("raw")
        expected = case.get("expected_status")
        if expected not in ("passed", "redacted", "withheld"):
            issues.append(Issue(path, f"invalid expected_status {expected!r}"))
            continue
        if not isinstance(raw, str):
            issues.append(Issue(path, "raw must be a string"))
            continue
        observed = set(secret_kinds(raw))
        expected_kinds = set(case.get("expect_kinds") or [])
        if not expected_kinds <= observed:
            issues.append(
                Issue(path, f"expected kinds {sorted(expected_kinds)} not matched; observed {sorted(observed)}")
            )
        if expected == "passed" and observed:
            issues.append(Issue(path, f"clean example matched {sorted(observed)}"))
        if expected == "withheld" and not (observed & withhold):
            issues.append(Issue(path, "withheld example did not match a withhold rule"))
        for forbidden in case.get("must_not_contain") or []:
            if forbidden not in raw:
                issues.append(
                    Issue(path, "must_not_contain value is not present in the raw example; fix the fixture")
                )
    return issues


def contract_health() -> dict:
    """Aggregate T01 contract checks (stdlib only, deterministic)."""
    checks: list[dict] = []
    issues = verify_checksums()
    checks.append({"check": "checksums", "passed": not issues})
    checks.extend({"check": "checksum-issue", "passed": False, "error": str(i)} for i in issues)

    structural = check_schema_structure()
    checks.append(
        {
            "check": "schema-structure",
            "passed": not structural,
            "errors": [str(i) for i in structural],
        }
    )
    surface = check_openapi_surface()
    checks.append(
        {"check": "openapi-surface", "passed": not surface, "errors": [str(i) for i in surface]}
    )
    catalog = check_error_catalog()
    checks.append(
        {"check": "error-catalog", "passed": not catalog, "errors": [str(i) for i in catalog]}
    )
    invariant_issues = check_invariants()
    checks.append(
        {
            "check": "semantic-invariants",
            "passed": not invariant_issues,
            "errors": [str(i) for i in invariant_issues],
        }
    )
    versions = version_consistency()
    checks.append(
        {"check": "versions", "passed": not versions, "errors": [str(i) for i in versions]}
    )
    policy_issues = check_redaction_policy()
    checks.append(
        {
            "check": "redaction-policy",
            "passed": not policy_issues,
            "errors": [str(i) for i in policy_issues],
        }
    )
    example_issues, example_report = shared_example_checks()
    checks.extend(example_report)
    checks.append(
        {
            "check": "shared-examples",
            "passed": not example_issues,
            "errors": [str(i) for i in example_issues],
        }
    )
    passed = all(check.get("passed") for check in checks)
    return {
        "scope": "frozen contract files, strict schema subset, shared examples, catalogs and digest helpers",
        "contract_version": CONTRACT_VERSION,
        "contracts_dir": str(contracts_dir()),
        "checks": checks,
        "passed": passed,
    }
