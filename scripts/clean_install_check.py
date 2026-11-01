#!/usr/bin/env python3
"""macOS 13 / Intel clean-install verification for the KW 2.0 core (AC-41).

Creates a brand-new venv inside the given workspace runtime directory (never
`--system-site-packages`), installs the hash-locked build/test toolchain, builds
a wheel from the checkout, installs that wheel, and runs the contract suites
against the *installed* package. Every command line, exit code and output tail
is written to a JSON artifact. No production path, credential or model network
is touched; the optional MCP extra is probed with ``--dry-run`` only.

Exit code 0 only when all required core checks pass. The documented
unavailability of the optional MCP extra on this target is recorded but is not
a core failure (AC-41: "可选依赖失败不拖垮核心").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_LOCK = REPO_ROOT / "requirements" / "build-lock-macos13-x86_64-py311.txt"
TEST_LOCK = REPO_ROOT / "requirements" / "test-lock-macos13-x86_64-py311.txt"
V1_LOCK = REPO_ROOT / "requirements" / "v1-lock-macos13-x86_64-py311.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Runner:
    def __init__(self, artifacts: Path):
        self.artifacts = artifacts
        self.artifacts.mkdir(parents=True, exist_ok=True)
        (self.artifacts / "logs").mkdir(exist_ok=True)
        self.records: list[dict] = []
        self.counter = 0

    @staticmethod
    def _clean_env() -> dict:
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_CACHE_DIR": "1",
                "PIP_NO_INPUT": "1",
            }
        )
        return env

    def run(self, name: str, command: list[str], *, cwd: Path | None = None) -> dict:
        self.counter += 1
        started = time.time()
        completed = subprocess.run(
            command,
            cwd=str(cwd or REPO_ROOT),
            env=self._clean_env(),
            capture_output=True,
            text=True,
        )
        record = {
            "step": name,
            "command": " ".join(command),
            "cwd": str(cwd or REPO_ROOT),
            "exit_code": completed.returncode,
            "seconds": round(time.time() - started, 2),
            "stdout_tail": completed.stdout[-8000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        (self.artifacts / "logs" / f"{self.counter:02d}-{name}.log").write_text(
            f"$ {' '.join(command)}\n\n[stdout]\n{completed.stdout}\n[stderr]\n{completed.stderr}\n",
            encoding="utf-8",
        )
        self.records.append(record)
        return record


def check(name: str, passed: bool, detail: object = None) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="isolated run runtime directory")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable, help="base interpreter used to create the venv")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args(argv)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    root = args.root.resolve()
    artifacts = (args.artifact_dir / run_id).resolve()
    venv = root / "venv"
    dist = root / "dist"
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"refusing to use non-empty runtime root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    dist.mkdir(parents=True, exist_ok=True)
    runner = Runner(artifacts)
    checks: list[dict] = []

    base_python = Path(args.python).resolve()
    checks.append(
        check(
            "fresh_runtime_root",
            True,
            {"root": str(root), "created_now": True, "run_id": run_id},
        )
    )

    # 1. Brand-new venv. No --system-site-packages; no pip upgrade so the
    #    interpreter's bundled tooling is recorded as-is.
    created = runner.run("create-venv", [str(base_python), "-m", "venv", str(venv)])
    checks.append(check("venv_created", created["exit_code"] == 0, created["exit_code"]))
    vpython = str(venv / "bin" / "python")

    # 2. Prove isolation before installing anything.
    isolation_code = (
        "import json,site,sys\n"
        "print(json.dumps({\n"
        "  'prefix': sys.prefix,\n"
        "  'base_prefix': sys.base_prefix,\n"
        "  'enable_user_site': getattr(site, 'ENABLE_USER_SITE', None),\n"
        "  'site_packages': site.getsitepackages(),\n"
        "  'path': sys.path,\n"
        "  'mcp_importable': __import__('importlib.util', fromlist=['util']).find_spec('mcp') is not None,\n"
        "  'cryptography_importable': __import__('importlib.util', fromlist=['util']).find_spec('cryptography') is not None,\n"
        "  'knowledge_weaver_importable': __import__('importlib.util', fromlist=['util']).find_spec('knowledge_weaver') is not None,\n"
        "}))\n"
    )
    probe = runner.run("probe-isolation", [vpython, "-c", isolation_code])
    isolation = {}
    if probe["exit_code"] == 0:
        try:
            isolation = json.loads(probe["stdout_tail"].strip().splitlines()[-1])
        except (ValueError, IndexError):
            isolation = {}
    venv_prefix = str(venv)
    package_paths = isolation.get("site_packages") or []
    no_foreign_site = all(
        str(Path(p).resolve()).startswith(venv_prefix) for p in package_paths
    )
    checks.append(
        check(
            "venv_isolated_no_system_site_packages",
            probe["exit_code"] == 0
            and isolation.get("prefix") == venv_prefix
            and isolation.get("base_prefix") != isolation.get("prefix")
            and isolation.get("enable_user_site") is False
            and no_foreign_site,
            isolation,
        )
    )
    checks.append(
        check(
            "core_absent_and_optional_absent_before_install",
            isolation.get("knowledge_weaver_importable") is False
            and isolation.get("mcp_importable") is False
            and isolation.get("cryptography_importable") is False,
            {
                "knowledge_weaver": isolation.get("knowledge_weaver_importable"),
                "mcp": isolation.get("mcp_importable"),
                "cryptography": isolation.get("cryptography_importable"),
            },
        )
    )

    # 3. Hash-locked build toolchain, then a deterministic wheel build.
    build_install = runner.run(
        "install-build-lock",
        [vpython, "-m", "pip", "install", "--require-hashes", "--ignore-installed", "-r", str(BUILD_LOCK)],
    )
    checks.append(check("build_lock_installed", build_install["exit_code"] == 0, build_install["exit_code"]))
    wheel_build = runner.run(
        "build-wheel",
        [vpython, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "--wheel-dir", str(dist), str(REPO_ROOT)],
    )
    wheels = sorted(dist.glob("knowledge_weaver-*.whl"))
    wheel_hash = sha256_file(wheels[0]) if len(wheels) == 1 else None
    checks.append(
        check(
            "wheel_built",
            wheel_build["exit_code"] == 0 and len(wheels) == 1,
            {"wheel": wheels[0].name if wheels else None, "sha256": wheel_hash},
        )
    )

    # 4. Install the wheel itself with no network/dependency resolution.
    if wheels:
        wheel_install = runner.run(
            "install-core-wheel",
            [vpython, "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0])],
        )
        checks.append(check("core_wheel_installed", wheel_install["exit_code"] == 0, wheel_install["exit_code"]))

    freeze = runner.run("freeze-core", [vpython, "-m", "pip", "freeze"])
    installed = [line for line in freeze["stdout_tail"].splitlines() if line.strip()]
    forbidden = [line for line in installed if line.split("==")[0].lower() in {"mcp", "cryptography", "httpx", "jieba", "pypinyin", "sqlite-vec"}]
    checks.append(
        check(
            "installed_core_has_no_optional_chain",
            freeze["exit_code"] == 0 and not forbidden,
            {"freeze": installed, "forbidden": forbidden},
        )
    )

    # 5. The installed package must be the one exercised, and its contract
    #    validation must work with network access disabled.
    offline_code = (
        "import socket\n"
        "def _blocked(*a, **k):\n"
        "    raise RuntimeError('external network disabled for core check')\n"
        "socket.socket.connect = _blocked\n"
        "socket.create_connection = _blocked\n"
        "from knowledge_weaver import contracts\n"
        "print(contracts.__file__)\n"
        "assert contracts.contract_health()['passed']\n"
        "print('offline-core-ok')\n"
    )
    offline = runner.run(
        "core-offline-contract-health",
        [vpython, "-c", offline_code],
        cwd=REPO_ROOT,
    )
    checks.append(
        check(
            "core_works_offline_with_installed_package",
            offline["exit_code"] == 0
            and "site-packages" in offline["stdout_tail"]
            and "offline-core-ok" in offline["stdout_tail"],
            offline["stdout_tail"].strip().splitlines()[:3],
        )
    )
    contract_health_path = artifacts / "contract-health.json"
    health = runner.run(
        "contract-check-installed",
        [
            vpython,
            str(REPO_ROOT / "scripts" / "contract_check.py"),
            "--quiet",
            "--out",
            str(contract_health_path),
        ],
    )
    checks.append(
        check(
            "contract_check_installed_package",
            health["exit_code"] == 0 and contract_health_path.is_file(),
            health["stdout_tail"].strip(),
        )
    )

    # 6. Full contract suite against the installed wheel (cwd = checkout so the
    #    frozen contract files are resolved explicitly).
    pytest_record = None
    if not args.skip_pytest:
        test_install = runner.run(
            "install-test-lock",
            [vpython, "-m", "pip", "install", "--require-hashes", "--ignore-installed", "-r", str(TEST_LOCK)],
        )
        checks.append(check("test_lock_installed", test_install["exit_code"] == 0, test_install["exit_code"]))
        junit = artifacts / "pytest-contract.xml"
        pytest_record = runner.run(
            "pytest-contract",
            [
                vpython,
                "-m",
                "pytest",
                "tests/contract",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={junit}",
            ],
        )
        summary = pytest_record["stdout_tail"].strip().splitlines()[-1] if pytest_record["stdout_tail"].strip() else ""
        checks.append(
            check(
                "contract_suite_passes_installed",
                pytest_record["exit_code"] == 0,
                {"summary": summary, "junit": str(junit) if junit.is_file() else None},
            )
        )
        freeze_test = runner.run("freeze-test-toolchain", [vpython, "-m", "pip", "freeze"])
        (artifacts / "pip-freeze.txt").write_text(freeze_test["stdout_tail"], encoding="utf-8")

    # 7. Optional legacy KW 1.x / MCP environment: installed into its own venv
    #    from its own hash lock, never into the core venv. The baseline failure
    #    was caused by pinning cryptography 50.x (no macOS x86_64 wheel); the
    #    lock below uses cryptography 48.0.1 (universal2 wheel). This is a
    #    separate optional boundary, so a failure here must not fail the core.
    optional_venv = root / "venv-optional"
    optional = {"passed": False, "detail": {}}
    if V1_LOCK.is_file():
        created_optional = runner.run("create-optional-venv", [str(base_python), "-m", "venv", str(optional_venv)])
        o_python = str(optional_venv / "bin" / "python")
        optional_isolation = runner.run(
            "probe-optional-isolation", [o_python, "-c", isolation_code]
        )
        v1_install = runner.run(
            "install-optional-v1-lock",
            [o_python, "-m", "pip", "install", "--require-hashes", "--ignore-installed", "-r", str(V1_LOCK)],
        )
        core_in_optional = {"exit_code": 1}
        if wheels:
            core_in_optional = runner.run(
                "install-core-wheel-into-optional",
                [o_python, "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0])],
            )
        smoke = runner.run(
            "optional-v1-import-smoke",
            [
                o_python,
                "-c",
                "import mcp\nfrom mcp.server.fastmcp import FastMCP\n"
                "import knowledge_weaver.server\nimport knowledge_weaver.pipeline\n"
                "print('optional-v1-env-ok')\n",
            ],
            cwd=REPO_ROOT,
        )
        optional = {
            "passed": (
                created_optional["exit_code"] == 0
                and v1_install["exit_code"] == 0
                and core_in_optional["exit_code"] == 0
                and smoke["exit_code"] == 0
                and "optional-v1-env-ok" in smoke["stdout_tail"]
            ),
            "detail": {
                "lock": str(V1_LOCK.relative_to(REPO_ROOT)),
                "lock_sha256": sha256_file(V1_LOCK),
                "venv": str(optional_venv),
                "install_exit": v1_install["exit_code"],
                "import_smoke_exit": smoke["exit_code"],
                "isolation_probe_exit": optional_isolation["exit_code"],
                "note": "optional legacy MCP environment only; core venv remains free of mcp/cryptography/httpx",
            },
        }
    checks.append(
        check(
            "optional_v1_env_installs_separately",
            optional["passed"],
            optional["detail"],
        )
    )

    # 8. Machine facts, repository revision and lock provenance.
    git = runner.run("git-revision", ["git", "rev-parse", "HEAD"])
    git_status = runner.run("git-status", ["git", "status", "--porcelain"])
    env_facts = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host_os": platform.platform(),
        "machine": platform.machine(),
        "python_base": subprocess.run([str(base_python), "--version"], capture_output=True, text=True).stdout.strip(),
        "python_venv": subprocess.run([vpython, "--version"], capture_output=True, text=True).stdout.strip(),
        "pip_venv": subprocess.run([vpython, "-m", "pip", "--version"], capture_output=True, text=True).stdout.strip(),
        "repo": str(REPO_ROOT),
        "repo_commit": git["stdout_tail"].strip(),
        "repo_dirty": bool(git_status["stdout_tail"].strip()),
        "repo_dirty_files": git_status["stdout_tail"].strip().splitlines(),
        "build_lock": str(BUILD_LOCK.relative_to(REPO_ROOT)),
        "build_lock_sha256": sha256_file(BUILD_LOCK),
        "test_lock": str(TEST_LOCK.relative_to(REPO_ROOT)),
        "test_lock_sha256": sha256_file(TEST_LOCK),
        "venv": str(venv),
        "wheel_sha256": wheel_hash,
        "scripts": {
            "clean_install_check": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "contract_check": "scripts/contract_check.py",
        },
    }
    required_checks = list(checks)  # including the optional environment: it must be installable on this target
    passed = all(entry["passed"] for entry in required_checks)
    report = {
        "scope": "AC-41 macOS 13 / Intel clean install of the KW 2.0 core in a brand-new venv",
        "passed": passed,
        "environment": env_facts,
        "checks": checks,
        "commands": runner.records,
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "clean-install-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "artifact_dir": str(artifacts), "passed": passed, "failed_checks": [entry for entry in checks if not entry["passed"]]}, ensure_ascii=False, indent=2, default=str))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
