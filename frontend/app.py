from __future__ import annotations

import json
import os

import httpx
import streamlit as st

from session_store import clear_session, load_session, save_session

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

ROLES = {
    "Manager": "manager",
    "Assistant Manager": "assistant_manager",
    "Developer": "developer",
}
ROLE_BY_VALUE = {value: label for label, value in ROLES.items()}

ROLE_HELP = {
    "manager": "PDF + Web",
    "assistant_manager": "PDF only",
    "developer": "Web only",
}


def api(method: str, path: str, *, timeout: float = 120.0, **kwargs) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=timeout) as client:
        return client.request(method, f"{API_URL}{path}", headers=headers, **kwargs)


def stream_query(question: str):
    headers = {"Accept": "text/event-stream"}
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{API_URL}/query/stream",
            headers=headers,
            json={"question": question},
        ) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")
                yield {"type": "error", "message": f"Request failed ({response.status_code}): {body}"}
                return
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue


def persist_ui_state() -> None:
    save_session(
        token=st.session_state.get("token"),
        username=st.session_state.get("username"),
        role=st.session_state.get("role"),
        allowed=st.session_state.get("allowed"),
        messages=st.session_state.get("messages"),
    )


def token_is_valid(token: str) -> bool:
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"{API_URL}/documents/all/visible",
                headers={"Authorization": f"Bearer {token}"},
            )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def restore_saved_session() -> None:
    saved = load_session()
    token = saved.get("token")
    if not token or not token_is_valid(token):
        if token:
            clear_session()
        return
    st.session_state.token = token
    st.session_state.username = saved.get("username")
    st.session_state.role = saved.get("role")
    st.session_state.allowed = saved.get("allowed") or []
    st.session_state.messages = saved.get("messages") or []


def render_citations(sources: list, key: str) -> None:
    if not sources:
        return
    labels = [
        f"{i}. {item.get('title') or 'Untitled'} · {item.get('locator') or ''}"
        for i, item in enumerate(sources, 1)
    ]
    chosen = st.selectbox("Citations", labels, key=key, index=0)
    item = sources[labels.index(chosen)]
    st.markdown(
        f"<div class='cite'><b>{item.get('title') or 'Untitled'}</b> · {item.get('locator') or ''}<br>{item.get('snippet') or ''}</div>",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Secure Multi-Source QA Agent", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; }
      .cite { font-size: 0.85rem; color: #334155; background: #f8fafc;
              border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "token" not in st.session_state:
    st.session_state.token = None
if "restored" not in st.session_state:
    restore_saved_session()
    st.session_state.restored = True

with st.sidebar:
    st.title("Access control")
    saved = load_session()
    default_username = st.session_state.get("username") or saved.get("username") or "demo.user"
    default_role_label = ROLE_BY_VALUE.get(
        st.session_state.get("role") or saved.get("role") or "manager",
        "Manager",
    )
    role_keys = list(ROLES.keys())
    role_index = role_keys.index(default_role_label) if default_role_label in role_keys else 0
    username = st.text_input("Username", value=default_username)
    role_label = st.selectbox("Role", role_keys, index=role_index)
    role = ROLES[role_label]
    st.caption(f"This role can use: **{ROLE_HELP[role]}**")

    if st.button("Issue JWT and sign in", type="primary", use_container_width=True):
        response = api("POST", "/auth/login", json={"username": username, "role": role})
        if response.status_code == 200:
            data = response.json()
            identity_changed = (
                st.session_state.get("username") != data["username"]
                or st.session_state.get("role") != data["role"]
            )
            st.session_state.token = data["access_token"]
            st.session_state.username = data["username"]
            st.session_state.role = data["role"]
            st.session_state.allowed = data["allowed_tools"]
            if identity_changed:
                st.session_state.messages = []
            persist_ui_state()
            st.success(f"Signed in as {data['username']} ({data['role']})")
        else:
            st.error(response.text)

    if st.session_state.token:
        st.code(st.session_state.token[:48] + "…", language=None)
        st.caption("JWT is sent as `Authorization: Bearer` on every API call.")
        if st.button("Sign out", use_container_width=True):
            st.session_state.token = None
            st.session_state.messages = []
            clear_session()
            st.rerun()

    st.divider()
    st.subheader("Sources")
    st.caption("PDFs and URLs are stored in Postgres and survive refresh/restart.")
    if not st.session_state.token:
        st.info("Sign in to upload sources.")
    else:
        allowed = set(st.session_state.get("allowed") or [])
        if "pdf_search" in allowed:
            uploaded = st.file_uploader("Upload PDF", type=["pdf"])
            if uploaded and st.button("Index PDF", use_container_width=True):
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                with st.spinner("Indexing PDF on the server (embeddings run on CPU — large files can take a few minutes)..."):
                    response = api("POST", "/documents", files=files, timeout=600.0)
                if response.status_code < 300:
                    rec = response.json()
                    st.success(f"Indexed {rec['title']} ({rec['pages']} pages, {rec['chunks']} chunks)")
                    st.rerun()
                else:
                    st.error(response.text)
        else:
            st.caption("PDF upload is hidden for this role.")

        if "web_search" in allowed:
            url = st.text_input("Website URL", placeholder="https://example.com")
            if st.button("Index website", use_container_width=True) and url:
                response = api("POST", "/websites", json={"url": url})
                if response.status_code < 300:
                    rec = response.json()
                    st.success(f"Indexed {rec['title']} ({rec['chunks']} chunks)")
                    st.rerun()
                else:
                    st.error(response.text)
        else:
            st.caption("Website indexing is hidden for this role.")

        visible = api("GET", "/documents/all/visible")
        if visible.status_code == 200:
            catalog = visible.json()
            pdfs = catalog.get("pdfs", [])
            websites = catalog.get("websites", [])
            if not pdfs and not websites:
                st.caption("No indexed sources yet.")
            for pdf in pdfs:
                cols = st.columns([4, 1])
                cols[0].write(f"📄 {pdf['title']} · {pdf.get('pages') or '?'} p.")
                if cols[1].button("✕", key=f"del-pdf-{pdf['id']}"):
                    api("DELETE", f"/documents/{pdf['id']}")
                    st.rerun()
            for site in websites:
                cols = st.columns([4, 1])
                url = site.get("url") or site.get("title")
                cols[0].write(f"🌐 {site['title']}")
                cols[0].caption(url)
                if cols[1].button("✕", key=f"del-web-{site['id']}"):
                    api("DELETE", f"/websites/{site['id']}")
                    st.rerun()
        elif visible.status_code == 401:
            st.warning("Session expired. Sign in again.")
            st.session_state.token = None
            clear_session()

st.title("Secure Multi-Source QA Agent")
st.caption("Answers are retrieved only from sources your JWT role is allowed to use.")

if not st.session_state.token:
    st.warning("Choose a role in the sidebar and issue a JWT to start chatting.")
else:
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_citations(message.get("sources") or [], key=f"cite-{index}")

    prompt = st.chat_input("Ask a question about the indexed PDFs or websites")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        persist_ui_state()
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            answer_box = st.empty()
            answer_box.markdown("_Retrieving sources…_")
            answer = ""
            sources: list = []
            tools_used: list[str] = []
            tools_allowed: list[str] = []
            llm_provider = "none"
            error_text = None
            for event in stream_query(prompt):
                kind = event.get("type")
                if kind == "meta":
                    sources = event.get("sources") or []
                    tools_used = event.get("tools_used") or []
                    tools_allowed = event.get("tools_allowed") or []
                    llm_provider = event.get("llm_provider") or "none"
                    answer_box.markdown("_Writing answer…_")
                elif kind == "token":
                    answer += event.get("text") or ""
                    answer_box.markdown(answer + " ▌")
                elif kind == "error":
                    error_text = event.get("message") or "Streaming failed."
                elif kind == "done":
                    answer = event.get("answer") or answer
            if error_text and not answer:
                answer_box.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text, "sources": []})
            else:
                answer_box.markdown(answer)
                meta = (
                    f"Tools used: `{', '.join(tools_used) or 'none'}` · "
                    f"Allowed: `{', '.join(tools_allowed)}` · "
                    f"LLM: `{llm_provider}`"
                )
                st.caption(meta)
                render_citations(sources, key=f"cite-live-{len(st.session_state.messages)}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer + "\n\n" + meta,
                        "sources": sources,
                    }
                )
            persist_ui_state()
