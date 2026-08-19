"""Website ingestion with a layered anti-bot strategy.

Order of attempts:
1. Optional ScraperAPI proxy (if SCRAPERAPI_KEY is set)
2. httpx with rotating desktop User-Agents
3. Playwright Chromium for JS / Cloudflare-style challenges

Content is cached on disk so re-indexing a URL does not always re-hit the site.
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from urllib.parse import quote, urlparse
from uuid import uuid4

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import WEB_CACHE_DIR, get_settings
from app.models.schemas import DocumentRecord
from app.retrieval.registry import add_record, find_web_by_url_hash
from app.retrieval.vectorstore import add_documents
from app.security import validate_public_http_url

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
]

_CLOUDFLARE_MARKERS = (
    "just a moment",
    "cf-browser-verification",
    "attention required",
    "checking your browser",
    "enable javascript and cookies",
)


def _splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _looks_blocked(status_code: int, text: str) -> bool:
    if status_code in {401, 403, 429, 503}:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _CLOUDFLARE_MARKERS)


def _html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return title, text.strip()


def _fetch_httpx(url: str) -> tuple[int, str]:
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
    with httpx.Client(follow_redirects=True, timeout=25.0, headers=headers) as client:
        response = client.get(url)
        return response.status_code, response.text


def _fetch_scraperapi(url: str) -> tuple[int, str]:
    key = get_settings().scraperapi_key
    if not key:
        raise RuntimeError("ScraperAPI key not configured")
    proxy_url = f"https://api.scraperapi.com/?api_key={key}&url={quote(url, safe='')}&render=true"
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        response = client.get(proxy_url)
        return response.status_code, response.text


def _fetch_playwright(url: str) -> tuple[int, str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is not installed") from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale="en-US",
            viewport={"width": 1366, "height": 768},
        )
        page = context.new_page()
        response = page.goto(url, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(2500)
        html = page.content()
        status = response.status if response is not None else 200
        context.close()
        browser.close()
        return status, html


def fetch_url(url: str) -> tuple[str, str, str]:
    """Return (final_url, title, text) using the cheapest successful strategy."""
    attempts: list[tuple[str, callable]] = [("httpx", _fetch_httpx)]
    if get_settings().scraperapi_key:
        attempts.insert(0, ("scraperapi", _fetch_scraperapi))
    attempts.append(("playwright", _fetch_playwright))

    last_error = "Unknown fetch error"
    for name, fetcher in attempts:
        try:
            status, html = fetcher(url)
        except Exception as exc:  # noqa: BLE001 - we want the next strategy
            last_error = f"{name} failed: {exc}"
            continue
        if _looks_blocked(status, html):
            last_error = f"{name} blocked with HTTP {status}"
            continue
        title, text = _html_to_text(html)
        if len(text) < 80:
            last_error = f"{name} returned too little text"
            continue
        return url, title or urlparse(url).netloc, text

    raise RuntimeError(
        f"Could not scrape {url}. Last error: {last_error}. "
        "Install Playwright browsers (`playwright install chromium`) "
        "or set SCRAPERAPI_KEY for protected sites."
    )


def ingest_website(raw_url: str) -> DocumentRecord:
    url = validate_public_http_url(str(raw_url))
    final_url, title, text = fetch_url(url)

    url_hash = hashlib.sha256(final_url.encode()).hexdigest()
    existing = find_web_by_url_hash(url_hash)
    if existing:
        existing_cache_path = (existing.extra or {}).get("cache_path")
        if existing_cache_path and Path(existing_cache_path).exists():
            return existing

    cache_name = hashlib.sha256(final_url.encode()).hexdigest()[:16]
    cache_path = WEB_CACHE_DIR / f"{cache_name}.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(f"{title}\n{final_url}\n\n{text}", encoding="utf-8")

    source_id = existing.id if existing else uuid4().hex
    documents: list[Document] = []
    for chunk_index, chunk in enumerate(_splitter().split_text(text)):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source_type": "web",
                    "source_id": source_id,
                    "title": title,
                    "url": final_url,
                    "chunk_index": chunk_index,
                },
            )
        )
    chunk_count = add_documents("web", documents)
    record = DocumentRecord(
        id=source_id,
        title=title,
        kind="web",
        url=final_url,
        chunks=chunk_count,
        status="ready",
        extra={"cache_path": str(cache_path), "url_hash": url_hash},
    )
    return add_record("web", record)
