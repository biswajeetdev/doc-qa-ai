import os
import io

import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from rag import build_index, chunk_text, retrieve, generate_answer

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

defaults = {"chunks": [], "index": None, "messages": [], "doc_name": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Ingestion helper ───────────────────────────────────────────────────────────

def ingest(text: str, name: str):
    chunks = chunk_text(text)
    if not chunks:
        st.sidebar.error("No text extracted. Try a different file.")
        return
    with st.spinner("Building vector index…"):
        index = build_index(chunks)
    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.doc_name = name
    st.session_state.messages = []
    st.rerun()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Setup")

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Used only for this session — never stored.",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.divider()
    st.subheader("Load document")

    tab_upload, tab_paste = st.tabs(["Upload PDF", "Paste text"])

    with tab_upload:
        pdf_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")
        if pdf_file and st.button("Process PDF", use_container_width=True):
            raw_text = "\n\n".join(
                page.extract_text() or ""
                for page in PdfReader(io.BytesIO(pdf_file.read())).pages
            )
            ingest(raw_text, pdf_file.name)

    with tab_paste:
        pasted = st.text_area("Paste document text", height=180, label_visibility="collapsed")
        if st.button("Process text", use_container_width=True) and pasted.strip():
            ingest(pasted, "pasted text")

    if st.session_state.doc_name:
        st.success(f"✅ **{st.session_state.doc_name}**")
        st.caption(f"{len(st.session_state.chunks)} chunks indexed")

    if st.button("🗑 Clear", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────

st.title("📄 DocQA")
st.caption("Upload a PDF or paste text — then ask anything about it.")

if not os.getenv("OPENAI_API_KEY"):
    st.info("👈 Enter your OpenAI API key in the sidebar to get started.")
    st.stop()

if st.session_state.index is None:
    st.info("👈 Upload a PDF or paste text in the sidebar to begin.")
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="score-badge">Chunk {src["idx"]+1} · '
                        f'similarity {src["score"]:.2f}</div>'
                        f'<div class="source-box">'
                        f'{src["chunk"][:420]}{"…" if len(src["chunk"]) > 420 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your document…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving & generating…"):
            retrieved = retrieve(prompt, st.session_state.index, st.session_state.chunks)
            answer = generate_answer(prompt, retrieved)
        st.markdown(answer)
        with st.expander("📎 Sources", expanded=False):
            for src in retrieved:
                st.markdown(
                    f'<div class="score-badge">Chunk {src["idx"]+1} · '
                    f'similarity {src["score"]:.2f}</div>'
                    f'<div class="source-box">'
                    f'{src["chunk"][:420]}{"…" if len(src["chunk"]) > 420 else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved,
    })
