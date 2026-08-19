from fastapi import APIRouter

from app.auth.jwt_handler import create_access_token
from app.auth.rbac import tools_for_role
from app.models.schemas import LoginRequest, TokenResponse
from app.security import sanitize_query

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest) -> TokenResponse:
    username = sanitize_query(body.username, max_chars=64)
    token, expires_in = create_access_token(username, body.role)
    return TokenResponse(
        access_token=token,
        role=body.role,
        username=username,
        expires_in=expires_in,
        allowed_tools=tools_for_role(body.role),
    )
