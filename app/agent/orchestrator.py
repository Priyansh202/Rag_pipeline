"""LangChain-backed agent that only receives tools the caller's JWT role allows."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import BaseTool

from app.agent.llm import chat, chat_stream, resolve_llm
from app.agent.tools.pdf_tool import pdf_search, search_pdfs
from app.agent.tools.web_tool import search_web, web_search
from app.auth.rbac import TOOL_PDF, TOOL_WEB, tools_for_role
from app.models.schemas import Citation, QueryResponse, Role

TOOL_REGISTRY: dict[str, BaseTool] = {
    TOOL_PDF: pdf_search,
    TOOL_WEB: web_search,
}

SYNTH_SYSTEM = """You are a careful enterprise question-answering agent.
Use ONLY the retrieved context. If the context is insufficient, say so.
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
            f"[{i}] type={item.source_type} title={item.title} locator={item.locator}\n{item.snippet[:300]}"
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


def retrieve_citations(question: str, role: Role) -> tuple[list[str], list[Citation]]:
    allowed = tools_for_role(role)
    selected = list(allowed)

    def _retrieve(name: str) -> list[Citation]:
        if name == TOOL_PDF:
            return search_pdfs(question)
        if name == TOOL_WEB:
            return search_web(question)
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
    return selected, unique


def run_agent(question: str, role: Role) -> QueryResponse:
    allowed = tools_for_role(role)
    selected, citations = retrieve_citations(question, role)

    handle = resolve_llm()
    reasoning = (
        f"Role={role.value} allowed={list(allowed)} selected={selected} "
        f"hits={len(citations)} llm={handle.provider}"
    )

    if not handle.available:
        answer = _extractive_answer(question, citations)
    else:
        user_prompt = (
            f"Question:\n{question}\n\nRetrieved context:\n{_format_context(citations)}\n\n"
            "Write a concise answer with inline citations."
        )
        try:
            answer = chat(
                [
                    {"role": "system", "content": SYNTH_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ]
            )
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


def stream_agent(question: str, role: Role) -> Iterator[dict]:
    """Yield SSE-friendly events: meta, token, done (or error)."""
    allowed = tools_for_role(role)
    selected, citations = retrieve_citations(question, role)
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

    user_prompt = (
        f"Question:\n{question}\n\nRetrieved context:\n{_format_context(citations)}\n\n"
        "Write a concise answer with inline citations."
    )
    pieces: list[str] = []
    try:
        for token in chat_stream(
            [
                {"role": "system", "content": SYNTH_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        ):
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
