"""Role-based access matrix.

Manager           PDF + Web
Assistant Manager PDF only
Developer         Web only

Enforced in three places so a single missed check cannot leak data:
1. API layer   – upload/list/delete routes require the matching permission
2. Agent layer – only permitted LangChain tools are bound into the agent
3. Index layer – PDF and web live in separate FAISS indexes
"""

from app.models.schemas import Role

TOOL_PDF = "pdf_search"
TOOL_WEB = "web_search"

ROLE_TOOLS: dict[Role, frozenset[str]] = {
    Role.MANAGER: frozenset({TOOL_PDF, TOOL_WEB}),
    Role.ASSISTANT_MANAGER: frozenset({TOOL_PDF}),
    Role.DEVELOPER: frozenset({TOOL_WEB}),
}


def tools_for_role(role: Role) -> list[str]:
    return sorted(ROLE_TOOLS[role])


def can_use(role: Role, tool_name: str) -> bool:
    return tool_name in ROLE_TOOLS[role]


def can_access_pdfs(role: Role) -> bool:
    return can_use(role, TOOL_PDF)


def can_access_web(role: Role) -> bool:
    return can_use(role, TOOL_WEB)
