from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class Role(str, Enum):
    MANAGER = "manager"
    ASSISTANT_MANAGER = "assistant_manager"
    DEVELOPER = "developer"


ROLE_LABELS = {
    Role.MANAGER: "Manager",
    Role.ASSISTANT_MANAGER: "Assistant Manager",
    Role.DEVELOPER: "Developer",
}


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    role: Role


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role
    username: str
    expires_in: int
    allowed_tools: list[str]


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list)


class Citation(BaseModel):
    source_type: str
    title: str
    locator: str
    snippet: str
    score: float | None = None
    url: str | None = None
    page: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Citation]
    role: Role
    tools_allowed: list[str]
    tools_used: list[str]
    llm_provider: str
    reasoning: str | None = None


class WebsiteCreate(BaseModel):
    url: HttpUrl


class DocumentRecord(BaseModel):
    id: str
    title: str
    kind: str
    pages: int | None = None
    url: str | None = None
    chunks: int = 0
    status: str = "ready"
    extra: dict[str, Any] = Field(default_factory=dict)


class RegistryResponse(BaseModel):
    pdfs: list[DocumentRecord]
    websites: list[DocumentRecord]


class MessageResponse(BaseModel):
    message: str
    id: str | None = None
