"""Start Docker databases, then the API and Streamlit UI."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _start_databases() -> None:
    print("Starting Docker Postgres (app db + pgvector)...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("Docker Compose failed. Is Docker Desktop running?")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for _ in range(30):
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "from app.db.wait import wait_for_databases; wait_for_databases(timeout_s=2)",
            ],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            print("Databases are ready.")
            return
        time.sleep(1)
    raise SystemExit("Timed out waiting for Postgres/pgvector.")


def main() -> None:
    _start_databases()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT,
        env=env,
    )
    time.sleep(1.5)
    ui = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "frontend" / "app.py"),
            "--server.port",
            "8501",
            "--server.headless",
            "true",
        ],
        cwd=ROOT,
        env=env,
    )
    print("API  -> http://127.0.0.1:8000/docs")
    print("UI   -> http://127.0.0.1:8501")
    try:
        ui.wait()
    finally:
        api.terminate()
        ui.terminate()


if __name__ == "__main__":
    main()
