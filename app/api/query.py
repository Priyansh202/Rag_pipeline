import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.agent.orchestrator import run_agent, stream_agent
from app.auth.dependencies import CurrentUser, get_current_user
from app.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse
from app.security import sanitize_query

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def ask(body: QueryRequest, user: CurrentUser = Depends(get_current_user)) -> QueryResponse:
    settings = get_settings()
    question = sanitize_query(body.question, max_chars=settings.max_query_chars)
    return run_agent(question, user.role, body.history)


@router.post("/stream")
def ask_stream(body: QueryRequest, user: CurrentUser = Depends(get_current_user)):
    settings = get_settings()
    question = sanitize_query(body.question, max_chars=settings.max_query_chars)

    def events():
        for event in stream_agent(question, user.role, body.history):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
