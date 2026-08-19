from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.jwt_handler import decode_access_token
from app.auth.rbac import can_access_pdfs, can_access_web
from app.models.schemas import Role

bearer_scheme = HTTPBearer(auto_error=False)


class CurrentUser:
    def __init__(self, username: str, role: Role):
        self.username = username
        self.role = role


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> CurrentUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token. Sign in first.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    username = payload.get("sub")
    role_value = payload.get("role")
    try:
        role = Role(role_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token contains an unknown role.",
        ) from exc
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing a subject.",
        )
    return CurrentUser(username=username, role=role)


def require_pdf_access(user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
    if not can_access_pdfs(user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{user.role.value}' cannot access PDF sources.",
        )
    return user


def require_web_access(user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
    if not can_access_web(user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{user.role.value}' cannot access web sources.",
        )
    return user
