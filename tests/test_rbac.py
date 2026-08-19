from fastapi.testclient import TestClient

from app.auth.rbac import can_access_pdfs, can_access_web
from app.main import app
from app.models.schemas import Role

client = TestClient(app)


def _token(role: str, username: str = "tester") -> str:
    response = client.post("/auth/login", json={"username": username, "role": role})
    assert response.status_code == 200
    return response.json()["access_token"]


def test_role_matrix():
    assert can_access_pdfs(Role.MANAGER) and can_access_web(Role.MANAGER)
    assert can_access_pdfs(Role.ASSISTANT_MANAGER) and not can_access_web(Role.ASSISTANT_MANAGER)
    assert not can_access_pdfs(Role.DEVELOPER) and can_access_web(Role.DEVELOPER)


def test_missing_token_is_rejected():
    response = client.post("/query", json={"question": "hello"})
    assert response.status_code == 401


def test_tampered_token_is_rejected():
    token = _token("manager")
    response = client.post(
        "/query",
        json={"question": "hello"},
        headers={"Authorization": f"Bearer {token}abc"},
    )
    assert response.status_code == 401


def test_developer_cannot_upload_pdf():
    token = _token("developer")
    response = client.post(
        "/documents",
        files={"file": ("notes.pdf", b"%PDF-1.4 dummy", "application/pdf")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_assistant_manager_cannot_add_website():
    token = _token("assistant_manager")
    response = client.post(
        "/websites",
        json={"url": "https://example.com"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_manager_query_reports_both_tools():
    token = _token("manager")
    response = client.post(
        "/query",
        json={"question": "What is the remote work policy?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body["tools_allowed"]) == {"pdf_search", "web_search"}


def test_developer_query_never_selects_pdf_tool():
    token = _token("developer")
    response = client.post(
        "/query",
        json={"question": "Summarize the employee handbook PDF"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tools_allowed"] == ["web_search"]
    assert "pdf_search" not in body["tools_used"]


def test_query_stream_emits_sse_events():
    token = _token("manager")
    with client.stream(
        "POST",
        "/query/stream",
        json={"question": "What is the remote work policy?"},
        headers={"Authorization": f"Bearer {token}"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "data:" in body
    assert '"type": "meta"' in body or '"type":"meta"' in body
    assert '"type": "token"' in body or '"type":"token"' in body or '"type": "done"' in body or '"type":"done"' in body
