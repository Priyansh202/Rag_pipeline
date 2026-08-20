"""LangChain-backed agent that only receives tools the caller's JWT role allows."""

from __future__ import annotations

import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import BaseTool

from app.agent.llm import chat, chat_stream, resolve_llm
from app.agent.memory import normalize_history, search_query_from_history
from app.agent.tools.pdf_tool import pdf_search, search_pdfs
from app.agent.tools.web_tool import search_web, web_search
from app.auth.rbac import TOOL_PDF, TOOL_WEB, tools_for_role
from app.config import get_settings
from app.models.schemas import ChatTurn, Citation, QueryResponse, Role

TOOL_REGISTRY: dict[str, BaseTool] = {
    TOOL_PDF: pdf_search,
    TOOL_WEB: web_search,
}

SYNTH_SYSTEM = """You are a careful enterprise question-answering agent.
Use ONLY the retrieved context plus the conversation history.
Read ALL passages before answering.
Prefer detailed narrative pages over table-of-contents or heading-only pages.
If later passages contain facts, use them — do not say details are unavailable.
Lead with a direct definition (what it is, where it is, size/type) when the context has it.
Resolve follow-up questions using prior turns (for example "how many units?" after "What is Alta Merita?").
Every non-trivial claim must cite a source using the locator already provided
(PDF title + page, or website URL).
Do not invent documents, pages, or URLs.
Do not mention access-control rules or other roles.
"""


def tools_for(role: Role) -> list[BaseTool]:
    return [TOOL_REGISTRY[name] for name in tools_for_role(role) if name in TOOL_REGISTRY]


def _format_context(citations: list[Citation]) -> str:
    if not citations:
        return "No context retrieved."
    blocks = []
    for i, item in enumerate(citations, 1):
        blocks.append(
            f"[{i}] type={item.source_type} title={item.title} locator={item.locator}\n{item.snippet}"
        )
    return "\n\n".join(blocks)


def _extractive_answer(question: str, citations: list[Citation]) -> str:
    if not citations:
        return (
            "I do not have indexed sources I am allowed to search for this question. "
            "Upload a permitted source and try again."
        )
    lines = [
        f"I could not reach an LLM, so here are the most relevant permitted passages for: {question}",
        "",
    ]
    for item in citations[:4]:
        lines.append(f"- {item.snippet} ({item.title}, {item.locator})")
    return "\n".join(lines)


_STOPWORDS = {"what", "is", "the", "a", "an", "of", "in", "to", "for", "on", "and", "or"}


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS and len(token) >= 4
    }


def _select_tools(question: str, allowed: list[str]) -> list[str]:
    """Skip website search when the question clearly names an indexed PDF."""
    selected = list(allowed)
    if TOOL_PDF not in selected or TOOL_WEB not in selected:
        return selected
    query_tokens = _content_tokens(question)
    if not query_tokens:
        return selected
    from app.retrieval.registry import list_sources

    for record in list_sources().pdfs:
        if query_tokens & _content_tokens(record.title):
            return [TOOL_PDF]
    return selected


def _llm_messages(question: str, citations: list[Citation], history: list[dict[str, str]]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYNTH_SYSTEM}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\nRetrieved context:\n{_format_context(citations)}\n\n"
                "Answer the current question using the retrieved context and prior conversation. "
                "Write a concise answer with inline citations."
            ),
        }
    )
    return messages


def retrieve_citations(
    question: str, role: Role, history: list[dict[str, str]] | None = None
) -> tuple[list[str], list[Citation]]:
    allowed = tools_for_role(role)
    history = history or []
    search_query = search_query_from_history(question, history)
    selected = _select_tools(search_query, list(allowed))

    def _retrieve(name: str) -> list[Citation]:
        if name == TOOL_PDF:
            return search_pdfs(search_query)
        if name == TOOL_WEB:
            return search_web(search_query)
        return []

    citations: list[Citation] = []
    if len(selected) <= 1:
        for name in selected:
            citations.extend(_retrieve(name))
    else:
        with ThreadPoolExecutor(max_workers=len(selected)) as ex:
            futures = {ex.submit(_retrieve, name): name for name in selected}
            results_by_name: dict[str, list[Citation]] = {name: [] for name in selected}
            for fut in futures:
                name = futures[fut]
                results_by_name[name] = fut.result()
            for name in selected:
                citations.extend(results_by_name.get(name) or [])

    seen: set[str] = set()
    unique: list[Citation] = []
    for item in citations:
        key = (item.source_type, item.title, item.locator, item.snippet[:80])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    unique.sort(key=lambda item: item.score or 0.0, reverse=True)
    return selected, unique


def run_agent(
    question: str, role: Role, history: list[ChatTurn] | None = None
) -> QueryResponse:
    settings = get_settings()
    turns = normalize_history(history, max_turns=settings.chat_history_turns)
    allowed = tools_for_role(role)
    selected, citations = retrieve_citations(question, role, turns)

    handle = resolve_llm()
    reasoning = (
        f"Role={role.value} allowed={list(allowed)} selected={selected} "
        f"hits={len(citations)} llm={handle.provider} history={len(turns)}"
    )

    if not handle.available:
        answer = _extractive_answer(question, citations)
    else:
        try:
            answer = chat(_llm_messages(question, citations, turns))
        except Exception as exc:  # noqa: BLE001
            answer = _extractive_answer(question, citations)
            reasoning += f" llm_error={exc}"

    return QueryResponse(
        answer=answer,
        sources=citations,
        role=role,
        tools_allowed=list(allowed),
        tools_used=selected,
        llm_provider=handle.provider,
        reasoning=reasoning,
    )


def stream_agent(
    question: str, role: Role, history: list[ChatTurn] | None = None
) -> Iterator[dict]:
    """Yield SSE-friendly events: meta, token, done (or error)."""
    settings = get_settings()
    turns = normalize_history(history, max_turns=settings.chat_history_turns)
    allowed = tools_for_role(role)
    selected, citations = retrieve_citations(question, role, turns)
    handle = resolve_llm()
    yield {
        "type": "meta",
        "role": role.value,
        "tools_allowed": list(allowed),
        "tools_used": selected,
        "llm_provider": handle.provider,
        "sources": [item.model_dump() for item in citations],
    }

    if not handle.available:
        answer = _extractive_answer(question, citations)
        for start in range(0, len(answer), 40):
            yield {"type": "token", "text": answer[start : start + 40]}
        yield {"type": "done", "answer": answer}
        return

    pieces: list[str] = []
    try:
        for token in chat_stream(_llm_messages(question, citations, turns)):
            pieces.append(token)
            yield {"type": "token", "text": token}
    except Exception as exc:  # noqa: BLE001
        answer = _extractive_answer(question, citations)
        if not pieces:
            for start in range(0, len(answer), 40):
                yield {"type": "token", "text": answer[start : start + 40]}
        yield {"type": "done", "answer": "".join(pieces) or answer, "error": str(exc)}
        return
    yield {"type": "done", "answer": "".join(pieces)}
