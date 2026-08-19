"""Local Docker image validation. Does not deploy to any cloud."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

IMAGE = "foresight-phase14:local"
CONTAINER = "foresight-phase14-sim"
HOST_PORT = 18000
RUNTIME_KEY = "phase14-docker-runtime-key"
EXPECTED_UCI_HASH = "331909f0fe191c0b9cb0418884b25eb59012f479f61f8b3e2ad51b729273e90d"
EXPECTED_SYN_HASH = "59a2b72024861d7f9c827596a52256af95facabfa796bdae5955374221cf1bf4"


def _run(args: list[str], timeout: int = 30, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    if log_path is None:
        return subprocess.run(
            args,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as handle:
        return subprocess.run(
            args,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )


def inspect_dockerfile() -> dict[str, Any]:
    docker_file = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    checks = {
        "dockerfile_exists": (PROJECT_ROOT / "Dockerfile").exists(),
        "non_root_user": "USER appuser" in docker_file,
        "uid_10001": "uid 10001" in docker_file or "--uid 10001" in docker_file,
        "healthcheck": "HEALTHCHECK" in docker_file and "/health" in docker_file,
        "production_env": "FORESIGHT_ENV=production" in docker_file,
        "auth_enabled": "FORESIGHT_API_AUTH_ENABLED=true" in docker_file,
        "no_api_key_in_image": "FORESIGHT_API_API_KEY" not in docker_file,
        "copies_models": "models/final/" in docker_file,
        "copies_registry": "final_model_registry.json" in docker_file,
        "dockerignore_env": ".env" in ignore,
        "dockerignore_venv": ".venv" in ignore,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def docker_daemon_available() -> tuple[bool, str]:
    try:
        proc = _run(["docker", "info"], timeout=20)
    except FileNotFoundError:
        return False, "docker CLI not installed"
    except subprocess.TimeoutExpired:
        return False, "docker info timed out"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, err[-1] if err else "docker daemon not running"
    return True, "ok"


def wait_for_daemon(timeout_s: int = 180) -> tuple[bool, str]:
    deadline = time.time() + timeout_s
    last = "docker daemon not running"
    while time.time() < deadline:
        ok, last = docker_daemon_available()
        if ok:
            return True, last
        time.sleep(3)
    return False, last


def _http(url: str, headers: dict[str, str] | None = None, timeout: int = 8) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except Exception as exc:
        return 0, str(exc)


def _wait_http(url: str, timeout_s: int = 60) -> tuple[int, str]:
    deadline = time.time() + timeout_s
    last = (0, "not ready")
    while time.time() < deadline:
        last = _http(url)
        if last[0] == 200:
            return last
        time.sleep(2)
    return last


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _image_exists(name: str) -> bool:
    proc = _run(["docker", "image", "inspect", name], timeout=20)
    return proc.returncode == 0


def _cleanup_container() -> None:
    _run(["docker", "rm", "-f", CONTAINER], timeout=30)


def run_docker_runtime() -> dict[str, Any]:
    skip = os.environ.get("FORESIGHT_SKIP_DOCKER", "").strip().lower() in {"1", "true", "yes"}
    if skip:
        return {
            "attempted": False,
            "passed": False,
            "status": "NOT IMPLEMENTED",
            "detail": "FORESIGHT_SKIP_DOCKER is set; runtime build skipped",
        }
    ok, detail = wait_for_daemon(180)
    if not ok:
        return {
            "attempted": True,
            "passed": False,
            "status": "NOT IMPLEMENTED",
            "detail": f"Docker daemon unavailable: {detail}",
        }
    if not _port_free(HOST_PORT):
        return {
            "attempted": True,
            "passed": False,
            "status": "FAIL",
            "detail": f"host port {HOST_PORT} is already in use",
        }

    _cleanup_container()
    build_log = PROJECT_ROOT / "outputs" / "phase14" / "docker_build.log"
    reuse_existing = _image_exists(IMAGE) and os.environ.get("FORESIGHT_DOCKER_REBUILD", "").strip().lower() not in {"1", "true", "yes"}
    built_this_run = False
    if reuse_existing:
        built_this_run = False
    else:
        try:
            build = _run(["docker", "build", "--pull=false", "-t", IMAGE, "."], timeout=1200, log_path=build_log)
        except subprocess.TimeoutExpired:
            if not _image_exists(IMAGE):
                return {
                    "attempted": True,
                    "passed": False,
                    "status": "FAIL",
                    "detail": "docker build timed out after 1200s",
                    "build": False,
                    "log": str(build_log),
                }
            build = None
        if build is not None and build.returncode != 0:
            tail = ""
            if build_log.exists():
                tail = build_log.read_text(encoding="utf-8", errors="replace")[-1500:]
            if not _image_exists(IMAGE):
                return {
                    "attempted": True,
                    "passed": False,
                    "status": "FAIL",
                    "detail": tail or "docker build failed",
                    "build": False,
                    "log": str(build_log),
                }
        else:
            built_this_run = True

    run = _run(
        [
            "docker", "run", "-d",
            "--name", CONTAINER,
            "-p", f"{HOST_PORT}:8000",
            "-e", f"FORESIGHT_API_API_KEY={RUNTIME_KEY}",
            IMAGE,
        ],
        timeout=60,
    )
    if run.returncode != 0:
        return {
            "attempted": True,
            "passed": False,
            "status": "FAIL",
            "detail": (run.stderr or run.stdout)[-1000:],
            "build": True,
            "run": False,
            "reused_local_image": reuse_existing,
        }

    evidence: dict[str, Any] = {
        "build": True,
        "run": True,
        "reused_local_image": reuse_existing,
        "built_this_run": built_this_run,
    }
    try:
        health_code, health_body = _wait_http(f"http://127.0.0.1:{HOST_PORT}/health", timeout_s=90)
        if health_code != 200:
            logs = _run(["docker", "logs", CONTAINER], timeout=20)
            evidence["container_logs"] = ((logs.stdout or "") + (logs.stderr or ""))[-2000:]
        ready_code, ready_body = (0, "")
        deadline = time.time() + 90
        while time.time() < deadline:
            ready_code, ready_body = _http(f"http://127.0.0.1:{HOST_PORT}/ready", timeout=20)
            if ready_code in {200, 503}:
                break
            time.sleep(2)
        denied_code, denied_body = _http(f"http://127.0.0.1:{HOST_PORT}/model")
        allowed_code, allowed_body = _http(
            f"http://127.0.0.1:{HOST_PORT}/model",
            headers={"X-API-Key": RUNTIME_KEY},
        )
        uid_proc = _run(["docker", "exec", CONTAINER, "id", "-u"], timeout=20)
        uid = (uid_proc.stdout or "").strip()
        inspect = _run(["docker", "inspect", IMAGE], timeout=30)
        env_blob = inspect.stdout.lower() if inspect.returncode == 0 else ""
        history = _run(["docker", "history", "--no-trunc", IMAGE], timeout=30)
        history_blob = (history.stdout or "").lower()
        hash_proc = _run(
            [
                "docker", "exec", CONTAINER, "python", "-c",
                "from src.forecasting.registry import load_registry, verify_hash; "
                "r=load_registry(); h={x['model_id']: verify_hash(x) for x in r}; "
                "print(h['uci_h1_phase8_lightgbm']); print(h['synthetic_h1_hurdle_th050'])",
            ],
            timeout=60,
        )
        hash_lines = [ln.strip() for ln in (hash_proc.stdout or "").splitlines() if ln.strip()]
        uci_hash = hash_lines[0] if len(hash_lines) >= 1 else ""
        syn_hash = hash_lines[1] if len(hash_lines) >= 2 else ""

        secret_free = (
            RUNTIME_KEY.lower() not in env_blob
            and RUNTIME_KEY.lower() not in history_blob
            and "password" not in history_blob
        )
        image_has_no_baked_key = "foresight_api_api_key" not in env_blob
        ready_ok = ready_code == 200 and '"status":"ready"' in ready_body.replace(" ", "")
        if not ready_ok:
            try:
                ready_ok = ready_code == 200 and json.loads(ready_body).get("status") == "ready"
            except Exception:
                ready_ok = False

        checks = {
            "health_200": health_code == 200,
            "ready_200": ready_ok,
            "unauth_401": denied_code == 401,
            "auth_200": allowed_code == 200,
            "non_root": uid not in {"", "0"},
            "uid_10001": uid == "10001",
            "no_secret_in_response": RUNTIME_KEY not in denied_body and RUNTIME_KEY not in allowed_body and RUNTIME_KEY not in health_body,
            "no_secret_in_image": secret_free and image_has_no_baked_key,
            "uci_hash": uci_hash == EXPECTED_UCI_HASH,
            "synthetic_hash": syn_hash == EXPECTED_SYN_HASH,
        }
        evidence.update({
            "health_code": health_code,
            "ready_code": ready_code,
            "denied_code": denied_code,
            "allowed_code": allowed_code,
            "uid": uid,
            "uci_hash": uci_hash,
            "synthetic_hash": syn_hash,
            "checks": checks,
        })
        passed = all(checks.values())
        return {
            "attempted": True,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
            "detail": json.dumps(checks),
            **evidence,
        }
    finally:
        _cleanup_container()


def run_docker_validation() -> dict[str, Any]:
    static = inspect_dockerfile()
    runtime = run_docker_runtime()
    # Runtime PASS is required for the Docker board item. Static-only is PARTIAL.
    if runtime.get("passed"):
        status = "PASS"
        passed = True
    elif static["passed"] and runtime.get("status") == "NOT IMPLEMENTED":
        status = "PARTIAL"
        passed = False
    else:
        status = "FAIL"
        passed = False
    return {
        "name": "Docker",
        "passed": passed,
        "status": status,
        "static": static,
        "runtime": runtime,
        "detail": f"static={static['status']} runtime={runtime.get('status')} {runtime.get('detail', '')}"[:500],
    }
