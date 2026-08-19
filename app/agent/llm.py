"""OpenAI-compatible LLM client. OpenRouter is the default when its key is set."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from app.config import get_settings


@dataclass
class LLMHandle:
    provider: str
    model: str
    base_url: str
    api_key: str

    @property
    def available(self) -> bool:
        return self.provider != "none"


def resolve_llm() -> LLMHandle:
    settings = get_settings()
    provider = settings.llm_provider.lower().strip()

    openrouter = LLMHandle(
        "openrouter",
        settings.openrouter_model,
        settings.openrouter_base_url,
        settings.openrouter_api_key,
    )
    openai = LLMHandle("openai", settings.openai_model, settings.openai_base_url, settings.openai_api_key)
    groq = LLMHandle(
        "groq",
        settings.groq_model,
        "https://api.groq.com/openai/v1",
        settings.groq_api_key,
    )
    ollama = LLMHandle("ollama", settings.ollama_model, settings.ollama_base_url, "ollama")
    none = LLMHandle("none", "extractive", "", "")

    chosen = {
        "none": none,
        "openai": openai if openai.api_key else none,
        "groq": groq if groq.api_key else none,
        "ollama": ollama if _ollama_up(settings.ollama_base_url) else none,
        "openrouter": openrouter if openrouter.api_key else none,
    }
    if provider in chosen:
        return chosen[provider]

    if openrouter.api_key:
        return openrouter
    if openai.api_key:
        return openai
    if groq.api_key:
        return groq
    if _ollama_up(settings.ollama_base_url):
        return ollama
    return none


def _ollama_up(base_url: str) -> bool:
    health = base_url.replace("/v1", "").rstrip("/") + "/api/tags"
    try:
        with httpx.Client(timeout=1.5) as client:
            return client.get(health).status_code == 200
    except httpx.HTTPError:
        return False


def chat(messages: list[dict[str, str]], temperature: float = 0.1) -> str:
    handle = resolve_llm()
    if not handle.available:
        raise RuntimeError("No LLM provider configured")
    settings = get_settings()
    headers = {"Content-Type": "application/json"}
    if handle.api_key and handle.api_key != "ollama":
        headers["Authorization"] = f"Bearer {handle.api_key}"
    if handle.provider == "openrouter":
        headers["HTTP-Referer"] = settings.openrouter_site_url
        headers["X-Title"] = settings.openrouter_app_name
    payload = {
        "model": handle.model,
        "messages": messages,
        "temperature": temperature,
    }
    url = handle.base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=90.0) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(
                f"{handle.provider} HTTP {response.status_code}: {response.text[:400]}"
            )
        data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _auth_headers(handle: LLMHandle) -> dict[str, str]:
    settings = get_settings()
    headers = {"Content-Type": "application/json"}
    if handle.api_key and handle.api_key != "ollama":
        headers["Authorization"] = f"Bearer {handle.api_key}"
    if handle.provider == "openrouter":
        headers["HTTP-Referer"] = settings.openrouter_site_url
        headers["X-Title"] = settings.openrouter_app_name
    return headers


def chat_stream(messages: list[dict[str, str]], temperature: float = 0.1) -> Iterator[str]:
    """Yield token deltas from an OpenAI-compatible streaming completion."""
    handle = resolve_llm()
    if not handle.available:
        raise RuntimeError("No LLM provider configured")
    payload = {
        "model": handle.model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    url = handle.base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=90.0) as client:
        with client.stream("POST", url, headers=_auth_headers(handle), json=payload) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", errors="replace")[:400]
                raise RuntimeError(f"{handle.provider} HTTP {response.status_code}: {body}")
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                else:
                    data = line.strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if delta:
                    yield delta


def choose_tools(question: str, allowed: list[str]) -> list[str]:
    if len(allowed) <= 1:
        return list(allowed)
    handle = resolve_llm()
    if not handle.available:
        return list(allowed)
    prompt = (
        "You route retrieval tools for a secure QA agent.\n"
        f"Allowed tools: {allowed}\n"
        'Return JSON only: {"tools": ["pdf_search" and/or "web_search"]}\n'
        "Pick every tool that could help. Default to all allowed tools if unsure.\n"
        f"Question: {question}"
    )
    try:
        raw = chat(
            [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start : end + 1])
        selected = [name for name in parsed.get("tools", []) if name in allowed]
        return selected or list(allowed)
    except Exception:  # noqa: BLE001
        return list(allowed)
