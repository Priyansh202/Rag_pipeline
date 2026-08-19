"""Pick API or Streamlit based on Railway service name or START_MODE."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _is_ui_service() -> bool:
    mode = os.environ.get("START_MODE", "").strip().lower()
    if mode in {"ui", "streamlit", "frontend"}:
        return True
    if mode in {"api", "backend"}:
        return False

    service_name = os.environ.get("RAILWAY_SERVICE_NAME", "").strip().lower()
    if "ui" in service_name or "streamlit" in service_name or "frontend" in service_name:
        return True
    return False


def main() -> None:
    target = ROOT / "scripts" / ("start_ui.py" if _is_ui_service() else "start_api.py")
    os.execv(sys.executable, [sys.executable, str(target)])


if __name__ == "__main__":
    main()
