#!/usr/bin/env python3
"""Validate the frozen DMA/KW contract v1 and print a machine-readable report.

Standard library only; safe to run in a bare venv on macOS 13 / Intel.
Exit code 0 only when every contract check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:  # installed package (clean-install verification) takes precedence
    from knowledge_weaver import contracts  # noqa: E402
except ImportError:  # source checkout without an install
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from knowledge_weaver import contracts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the JSON report to this path")
    parser.add_argument("--quiet", action="store_true", help="only print pass/fail")
    args = parser.parse_args(argv)
    try:
        result = contracts.contract_health()
    except contracts.ContractError as exc:
        result = {
            "scope": "frozen contract validation",
            "passed": False,
            "error": str(exc),
            "checks": [],
        }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.quiet:
        print("contract-check:", "PASS" if result["passed"] else "FAIL")
    else:
        print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
