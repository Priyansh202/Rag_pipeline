"""Input sanitization and SSRF guards used by upload / scrape paths."""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

from fastapi import HTTPException, status

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_PRIVATE_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def sanitize_filename(name: str) -> str:
    cleaned = Path(name).name
    cleaned = _SAFE_FILENAME.sub("_", cleaned).strip("._")
    stem = cleaned[:-4] if cleaned.lower().endswith(".pdf") else cleaned
    stem = stem.strip("._") or "document"
    if len(stem) > 176:
        stem = stem[:176]
    return f"{stem}.pdf"


def sanitize_query(text: str, max_chars: int = 2000) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query is empty.")
    if len(cleaned) > max_chars:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query exceeds {max_chars} characters.",
        )
    return cleaned


def validate_public_http_url(raw: str) -> str:
    parsed = urlparse(raw.strip())
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only http(s) URLs are allowed.",
        )
    host = (parsed.hostname or "").lower()
    if not host or host in _PRIVATE_HOSTS or host.endswith(".local"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Internal or local URLs are not allowed.",
        )
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not resolve host '{host}'.",
        ) from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL resolves to a private or reserved address.",
            )
    return parsed.geturl()
