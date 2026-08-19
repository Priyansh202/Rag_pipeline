import pytest
from fastapi import HTTPException

from app.security import sanitize_filename, sanitize_query, validate_public_http_url


def test_filename_strips_paths_and_forces_pdf():
    assert sanitize_filename("../../etc/passwd") == "passwd.pdf"
    assert sanitize_filename("My Report (v2).PDF") == "My_Report_v2.pdf"


def test_query_rejects_blank_and_oversize():
    with pytest.raises(HTTPException):
        sanitize_query("   ")
    with pytest.raises(HTTPException):
        sanitize_query("x" * 50, max_chars=10)


def test_ssrf_blocks_localhost():
    with pytest.raises(HTTPException):
        validate_public_http_url("http://127.0.0.1/secret")
    with pytest.raises(HTTPException):
        validate_public_http_url("file:///etc/passwd")
