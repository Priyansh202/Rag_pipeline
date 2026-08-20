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
    "manager": "PDFs and websites",
    "assistant_manager": "PDFs only",
    "developer": "Websites only",
}

SUGGESTIONS = [
    "What is Alta Merita?",
    "How many units are there?",
    "Summarize the offering",
    "Where is the property located?",
]


def api(method: str, path: str, *, timeout: float = 120.0, **kwargs) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=timeout) as client:
        return client.request(method, f"{API_URL}{path}", headers=headers, **kwargs)


def history_payload() -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for message in st.session_state.get("messages") or []:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if role == "assistant":
            content = content.split("\n\nTools used:")[0].strip()
        if not content:
            continue
        turns.append({"role": role, "content": content[:1500]})
    return turns[-8:]


def stream_query(question: str, history: list[dict[str, str]] | None = None):
    headers = {"Accept": "text/event-stream"}
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{API_URL}/query/stream",
            headers=headers,
            json={"question": question, "history": history if history is not None else history_payload()},
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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          footer { visibility: hidden; }
          [data-testid="stToolbar"], .stAppDeployButton, [data-testid="stDecoration"] { display: none !important; }
          .block-container { padding-top: 3.2rem; padding-bottom: 2rem; max-width: 900px; }
          [data-testid="stHeader"] { background: #0B1220; }
          .hero { padding-top: 0.4rem; overflow: visible; }
          .hero h1 {
            font-size: 1.7rem;
            letter-spacing: -0.02em;
            line-height: 1.35;
            margin: 0 0 0.4rem 0;
            overflow: visible;
          }
          [data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            color: #EEF2FF !important;
            background: #4F46E5 !important;
            border-radius: 10px !important;
            width: 2.4rem !important;
            height: 2.4rem !important;
            margin: 0.45rem 0.5rem !important;
          }
          [data-testid="stSidebarCollapseButton"] {
            color: #E8EEF8 !important;
          }
          [data-testid="stSidebar"] {
            background: #101827;
            border-right: 1px solid #243044;
            min-width: 340px !important;
            width: 340px !important;
            transform: none !important;
            visibility: visible !important;
            display: block !important;
          }
          [data-testid="stSidebarCollapseButton"] { display: none !important; }
          .nav-wrap {
            background: #151C2C; border: 1px solid #243044; border-radius: 14px;
            padding: 0.75rem 1rem 0.35rem; margin-bottom: 1.1rem;
          }
          .nav-title { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }
          .nav-sub { color: #94A3B8; font-size: 0.82rem; margin: 0.15rem 0 0.4rem; }
          div.stButton > button {
            background: #312E81; color: #EEF2FF; border: 1px solid #6366F1;
            border-radius: 10px; height: 2.4rem;
          }
          div.stButton > button:hover { background: #4338CA; color: #fff; border-color: #818CF8; }
          .hero p { color: #94A3B8; margin-top: 0; line-height: 1.45; }
          .chip-row { display: flex; gap: 0.4rem; flex-wrap: wrap; margin: 0.4rem 0 1rem; }
          .chip {
            display: inline-block; font-size: 0.75rem; color: #C7D2FE;
            background: #1E1B4B; border: 1px solid #3730A3; border-radius: 999px;
            padding: 0.2rem 0.65rem;
          }
          .source-card {
            background: #151C2C; border: 1px solid #243044; border-radius: 10px;
            padding: 0.55rem 0.7rem; margin-bottom: 0.45rem;
          }
          .source-card small { color: #94A3B8; }
          .cite {
            font-size: 0.84rem; color: #CBD5E1; background: #111827;
            border: 1px solid #243044; border-radius: 10px;
            padding: 0.7rem 0.85rem; margin: 0.35rem 0;
            line-height: 1.45;
          }
          .cite b { color: #E8EEF8; }
          .empty {
            text-align: center; padding: 2.4rem 1rem 1.2rem;
            border: 1px dashed #334155; border-radius: 16px; background: #101827;
          }
          .empty h3 { margin-bottom: 0.35rem; }
          .empty p { color: #94A3B8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_meta_pills(tools_used: list[str], llm_provider: str) -> None:
    used = ", ".join(tools_used) if tools_used else "none"
    st.markdown(
        f'<div class="chip-row">'
        f'<span class="chip">{used}</span>'
        f'<span class="chip">{llm_provider}</span>'
        f'<span class="chip">memory on</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_citations(sources: list, key: str) -> None:
    if not sources:
        return
    with st.expander(f"{len(sources)} source{'s' if len(sources) != 1 else ''}", expanded=False):
        for i, item in enumerate(sources, 1):
            title = item.get("title") or "Untitled"
            locator = item.get("locator") or ""
            snippet = (item.get("snippet") or "").strip()
            kind = item.get("source_type") or "pdf"
            icon = "📄" if kind == "pdf" else "🌐"
            st.markdown(
                f"<div class='cite'><b>{icon} {i}. {title}</b> · {locator}<br>{snippet}</div>",
                unsafe_allow_html=True,
            )


def run_turn(prompt: str) -> None:
    history_before = history_payload()
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    persist_ui_state()
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        answer_box = st.empty()
        answer_box.markdown("_Searching your sources…_")
        answer = ""
        sources: list = []
        tools_used: list[str] = []
        llm_provider = "none"
        error_text = None
        for event in stream_query(prompt, history_before):
            kind = event.get("type")
            if kind == "meta":
                sources = event.get("sources") or []
                tools_used = event.get("tools_used") or []
                llm_provider = event.get("llm_provider") or "none"
                answer_box.markdown("_Writing…_")
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
            render_meta_pills(tools_used, llm_provider)
            render_citations(sources, key=f"cite-live-{len(st.session_state.messages)}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "tools_used": tools_used,
                    "llm_provider": llm_provider,
                }
            )
        persist_ui_state()


st.set_page_config(
    page_title="Source Chat",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "token" not in st.session_state:
    st.session_state.token = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "restored" not in st.session_state:
    restore_saved_session()
    st.session_state.restored = True

with st.sidebar:
    st.markdown("### Source Chat")
    st.caption("Ask your PDFs and sites. Answers stay inside your role.")

    saved = load_session()
    default_username = st.session_state.get("username") or saved.get("username") or "demo.user"
    default_role_label = ROLE_BY_VALUE.get(
        st.session_state.get("role") or saved.get("role") or "manager",
        "Manager",
    )
    role_keys = list(ROLES.keys())
    role_index = role_keys.index(default_role_label) if default_role_label in role_keys else 0

    with st.expander("Sign in", expanded=not st.session_state.token):
        username = st.text_input("Name", value=default_username, key="sidebar-username")
        role_label = st.selectbox("Role", role_keys, index=role_index, key="sidebar-role")
        role = ROLES[role_label]
        st.caption(ROLE_HELP[role])
        if st.button("Continue", type="primary", use_container_width=True):
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
                st.rerun()
            else:
                st.error(response.text)

    if st.session_state.token:
        who = st.session_state.get("username") or "user"
        role_name = ROLE_BY_VALUE.get(st.session_state.get("role") or "", "Role")
        st.markdown(
            f'<div class="chip-row"><span class="chip">{who}</span>'
            f'<span class="chip">{role_name}</span></div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        if c1.button("New chat", use_container_width=True):
            st.session_state.messages = []
            persist_ui_state()
            st.rerun()
        if c2.button("Sign out", use_container_width=True):
            st.session_state.token = None
            st.session_state.messages = []
            clear_session()
            st.rerun()

    st.divider()
    st.markdown("### Index sources")
    st.caption("Upload a PDF or paste a website URL.")
    if not st.session_state.token:
        st.caption("Sign in first.")
    else:
        allowed = set(st.session_state.get("allowed") or [])
        if "pdf_search" in allowed:
            uploaded = st.file_uploader("PDF file", type=["pdf"], key="side-pdf")
            if uploaded and st.button("Index PDF", use_container_width=True, type="primary", key="side-index-pdf"):
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                with st.spinner("Indexing PDF…"):
                    response = api("POST", "/documents", files=files, timeout=600.0)
                if response.status_code < 300:
                    rec = response.json()
                    st.success(f"{rec['title']} · {rec['pages']} p. · {rec['chunks']} chunks")
                    st.rerun()
                else:
                    st.error(response.text)
        else:
            st.caption("This role cannot index PDFs.")

        if "web_search" in allowed:
            url = st.text_input("Website URL", placeholder="https://example.com", key="side-url")
            if st.button("Index website", use_container_width=True, key="side-index-web") and url:
                with st.spinner("Indexing website…"):
                    response = api("POST", "/websites", json={"url": url})
                if response.status_code < 300:
                    rec = response.json()
                    st.success(f"{rec['title']} · {rec['chunks']} chunks")
                    st.rerun()
                else:
                    st.error(response.text)
        else:
            st.caption("This role cannot index websites.")

        visible = api("GET", "/documents/all/visible")
        if visible.status_code == 200:
            catalog = visible.json()
            pdfs = catalog.get("pdfs", [])
            websites = catalog.get("websites", [])
            st.caption(f"{len(pdfs)} PDF(s) · {len(websites)} site(s)")
            for pdf in pdfs:
                cols = st.columns([5, 1])
                cols[0].markdown(f"📄 {pdf['title']}")
                cols[0].caption(f"{pdf.get('pages') or '?'} pages · {pdf.get('chunks') or 0} chunks")
                if cols[1].button("✕", key=f"del-pdf-{pdf['id']}"):
                    api("DELETE", f"/documents/{pdf['id']}")
                    st.rerun()
            for site in websites:
                cols = st.columns([5, 1])
                cols[0].markdown(f"🌐 {site['title']}")
                cols[0].caption(site.get("url") or "")
                if cols[1].button("✕", key=f"del-web-{site['id']}"):
                    api("DELETE", f"/websites/{site['id']}")
                    st.rerun()
        elif visible.status_code == 401:
            st.warning("Session expired.")
            st.session_state.token = None
            clear_session()
            st.rerun()

st.markdown(
    '<div class="hero"><h1>Ask your sources</h1>'
    "<p>Cited answers from indexed PDFs and websites. Follow-ups keep chat memory.</p></div>",
    unsafe_allow_html=True,
)

if not st.session_state.token:
    st.markdown(
        '<div class="empty"><h3>Sign in to chat</h3>'
        "<p>Use <b>Sign in</b> in the left sidebar. Choose a name and role, then Continue.</p></div>",
        unsafe_allow_html=True,
    )
else:
    if not st.session_state.messages:
        st.markdown(
            '<div class="empty"><h3>Start a conversation</h3>'
            "<p>Try a suggested question, or type your own below.</p></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, suggestion in enumerate(SUGGESTIONS):
            if cols[i % 2].button(suggestion, use_container_width=True, key=f"sug-{i}"):
                st.session_state.pending_prompt = suggestion
                st.rerun()

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_meta_pills(message.get("tools_used") or [], message.get("llm_provider") or "openrouter")
                render_citations(message.get("sources") or [], key=f"cite-{index}")

    prompt = st.chat_input("Ask about your indexed sources…")
    pending = st.session_state.pending_prompt
    if pending:
        st.session_state.pending_prompt = None
        run_turn(pending)
    elif prompt:
        run_turn(prompt)
