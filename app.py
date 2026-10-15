import os
import io
import html

import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from rag import build_index, chunk_text, retrieve, generate_answer, make_chat_client

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocQA — Ask your documents",
    page_icon="📄",
    layout="centered",
)

st.markdown("""
<style>
    .source-box {
        background: #f6f8fa;
        border-left: 3px solid #0969da;
        padding: 0.5rem 0.75rem;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #444;
        margin-top: 0.3rem;
    }
    .score-badge {
        font-size: 0.72rem;
        color: #6e7781;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────

defaults = {
    "chunks": [],
    "index": None,
    "messages": [],
    "doc_name": None,
    "chat_client": None,
    "provider": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────────────────

def ingest(text: str, name: str, provider: str, chat_client, openai_client=None):
    chunks = chunk_text(text)
    if not chunks:
        st.sidebar.error("No text extracted. Try a different file.")
        return
    with st.spinner("Building vector index..."):
        index = build_index(chunks, provider=provider, openai_client=openai_client)
    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.doc_name = name
    st.session_state.messages = []
    st.session_state.chat_client = chat_client
    st.session_state.provider = provider
    st.rerun()


def render_sources(sources: list) -> None:
    with st.expander("Sources", expanded=False):
        for src in sources:
            safe_chunk = html.escape(src["chunk"][:420])
            ellipsis = "&#8230;" if len(src["chunk"]) > 420 else ""
            st.markdown(
                f'<div class="score-badge">Chunk {src["idx"]+1} · '
                f'similarity {src["score"]:.2f}</div>'
                f'<div class="source-box">{safe_chunk}{ellipsis}</div>',
                unsafe_allow_html=True,
            )

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Setup")

    provider_label = st.selectbox(
        "LLM Provider",
        ["Groq (free)", "OpenAI"],
        help="Groq is free — no credit card needed. Embeddings always run locally.",
    )
    provider = "groq" if "Groq" in provider_label else "openai"

    if provider == "groq":
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            value=os.getenv("GROQ_API_KEY", ""),
        ).strip()
        st.caption("Free key at [console.groq.com/keys](https://console.groq.com/keys)")
        chat_client = make_chat_client(api_key, "groq") if api_key else None
        openai_client = None
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            value=os.getenv("OPENAI_API_KEY", ""),
        ).strip()
        chat_client = make_chat_client(api_key, "openai") if api_key else None
        openai_client = chat_client

    ready = bool(api_key)

    st.divider()
    st.subheader("Load document")

    tab_upload, tab_paste = st.tabs(["Upload PDF", "Paste text"])

    with tab_upload:
        uploaded = st.file_uploader("Choose a PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")
        if uploaded and st.button("Process file", use_container_width=True, disabled=not ready):
            MAX_FILE_MB = 10
            if uploaded.size > MAX_FILE_MB * 1024 * 1024:
                st.error(f"File too large. Maximum size is {MAX_FILE_MB} MB.")
            elif uploaded.type == "text/plain" or uploaded.name.endswith(".txt"):
                raw_text = uploaded.read().decode("utf-8", errors="replace")
                ingest(raw_text, uploaded.name, provider, chat_client, openai_client)
            else:
                raw_text = "\n\n".join(
                    page.extract_text() or ""
                    for page in PdfReader(io.BytesIO(uploaded.read())).pages
                )
                ingest(raw_text, uploaded.name, provider, chat_client, openai_client)

    with tab_paste:
        pasted = st.text_area("Paste document text", height=180, label_visibility="collapsed")
        if st.button("Process text", use_container_width=True, disabled=not ready) and pasted.strip():
            ingest(pasted, "pasted text", provider, chat_client, openai_client)

    if st.session_state.doc_name:
        st.success(f"**{st.session_state.doc_name}**")
        st.caption(f"{len(st.session_state.chunks)} chunks indexed")

    if st.button("Clear", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────

st.title("DocQA")
st.caption("Upload a PDF or paste text — then ask anything about it.")

if not ready:
    provider_hint = "Groq" if provider == "groq" else "OpenAI"
    st.info(f"Enter your {provider_hint} API key in the sidebar to get started.")
    st.stop()

if st.session_state.index is None:
    st.info("Upload a PDF or paste text in the sidebar to begin.")
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved = retrieve(
                prompt,
                st.session_state.index,
                st.session_state.chunks,
                provider=st.session_state.provider,
                openai_client=openai_client if st.session_state.provider == "openai" else None,
            )
            answer = generate_answer(
                prompt,
                retrieved,
                st.session_state.chat_client,
                provider=st.session_state.provider,
            )
        st.markdown(answer)
        render_sources(retrieved)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved,
    })
