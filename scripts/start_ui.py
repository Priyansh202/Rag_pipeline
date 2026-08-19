"""Bind Streamlit to Railway's $PORT (or 8501 locally)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    os.chdir(ROOT)
    os.environ.setdefault("PYTHONPATH", str(ROOT))
    port = os.environ.get("PORT", "8501")
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "frontend" / "app.py"),
            "--server.port",
            str(port),
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
            "--browser.gatherUsageStats",
            "false",
        ],
    )


if __name__ == "__main__":
    main()
