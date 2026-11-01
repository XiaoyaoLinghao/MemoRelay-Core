"""External extractor command client (T17).

The semantic model may be external (PRD §9). This adapter sends only the
sanitized input group on stdin to a configured local command and accepts its
JSON proposal on stdout:

* credentials stay in the command/host environment; they are never put in CLI
  arguments, prompts or logs (the command is responsible for its own auth);
* a non-zero exit, timeout, invalid JSON or schema-invalid proposal is a
  retryable model failure; the raw source stays durable;
* the proposal is still validated by the extraction pipeline (evidence,
  authority, project, secret re-check), so this adapter cannot grant trust.
"""

from __future__ import annotations

import json
import subprocess

from .proposals import ExtractionError


class CommandExtractor:
    def __init__(self, command: list[str], *, timeout: float = 120.0) -> None:
        if not command or not isinstance(command, list):
            raise ValueError("command must be a non-empty argument list")
        self.command = list(command)
        self.timeout = float(timeout)

    def extract(self, parts, context: dict) -> dict:
        payload = {
            "schema_version": "1.0",
            "context": context,
            "parts": [
                {
                    "event_id": part.event_id,
                    "event_revision": part.event_revision,
                    "part_index": part.part_index,
                    "text": part.text,
                    "role": part.role,
                    "origin_kind": part.origin_kind,
                    "event_time": part.event_time,
                }
                for part in parts
            ],
        }
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise ExtractionError("MODEL_UNAVAILABLE", "external extractor timed out", retryable=True) from None
        except OSError as exc:
            raise ExtractionError(
                "MODEL_UNAVAILABLE", f"external extractor failed to start: {type(exc).__name__}", retryable=True
            ) from None
        if completed.returncode != 0:
            raise ExtractionError(
                "MODEL_UNAVAILABLE",
                f"external extractor exited with {completed.returncode}",
                retryable=True,
            )
        try:
            proposal = json.loads(completed.stdout)
        except json.JSONDecodeError:
            raise ExtractionError(
                "MODEL_UNAVAILABLE", "external extractor returned invalid JSON", retryable=True
            ) from None
        if not isinstance(proposal, dict):
            raise ExtractionError(
                "MODEL_UNAVAILABLE", "external extractor returned a non-object proposal", retryable=True
            )
        return proposal
