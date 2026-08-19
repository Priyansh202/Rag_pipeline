from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "pdfs" / "sample_acme_handbook.pdf"
client = TestClient(app)


def test_pdf_upload_and_page_citation():
    if not SAMPLE.exists():
        pytest.skip("sample PDF is missing; run scripts/generate_sample_pdf.py")
    token = client.post("/auth/login", json={"username": "e2e", "role": "manager"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post(
        "/documents",
        files={"file": ("sample_acme_handbook.pdf", SAMPLE.read_bytes(), "application/pdf")},
        headers=headers,
    )
    assert response.status_code == 201
    body = response.json()
    assert body["pages"] == 3
    assert body["chunks"] >= 1

    asked = client.post(
        "/query",
        json={"question": "How many remote days does ACME allow?"},
        headers=headers,
    )
    assert asked.status_code == 200
    payload = asked.json()
    assert payload["sources"]
    assert payload["sources"][0]["page"] == 1
    assert "three days" in payload["sources"][0]["snippet"].lower()
    assert "pdf_search" in payload["tools_used"]
