import os
from typing import Optional

import faiss
import numpy as np
import tiktoken
from openai import OpenAI
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 40
TOP_K = 4

_enc = tiktoken.get_encoding("cl100k_base")
_embedder: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def make_chat_client(api_key: str, provider: str = "groq") -> OpenAI:
    if provider == "openai":
        return OpenAI(api_key=api_key)
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


# ── Text chunking ──────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


# ── Embeddings & index ─────────────────────────────────────────────────────────

def _embed_local(texts: list[str]) -> np.ndarray:
    return _get_embedder().encode(texts, normalize_embeddings=True).astype("float32")


def _embed_openai(texts: list[str], client: OpenAI) -> np.ndarray:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    vecs = np.array([item.embedding for item in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


def build_index(
    chunks: list[str],
    provider: str = "groq",
    openai_client: Optional[OpenAI] = None,
) -> faiss.IndexFlatIP:
    vecs = (
        _embed_openai(chunks, openai_client)
        if provider == "openai" and openai_client
        else _embed_local(chunks)
    )
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[str],
    provider: str = "groq",
    openai_client: Optional[OpenAI] = None,
) -> list[dict]:
    q_vec = (
        _embed_openai([query], openai_client)
        if provider == "openai" and openai_client
        else _embed_local([query])
    )
    scores, indices = index.search(q_vec, TOP_K)
    return [
        {"chunk": chunks[idx], "score": float(score), "idx": int(idx)}
        for score, idx in zip(scores[0], indices[0])
        if idx != -1
    ]


# ── Generation ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document assistant. Answer the user's question
using ONLY the context provided. If the answer is not in the context, say so clearly.
Keep answers concise and cite the relevant passage when helpful."""


def generate_answer(query: str, retrieved: list[dict], client: OpenAI, provider: str = "groq") -> str:
    if not retrieved:
        return "No relevant content found in the document for that question."
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]: {r['chunk']}" for i, r in enumerate(retrieved)
    )
    model = OPENAI_CHAT_MODEL if provider == "openai" else GROQ_CHAT_MODEL
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
