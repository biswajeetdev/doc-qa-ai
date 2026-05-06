import os
import math
import textwrap
from typing import Optional

import faiss
import numpy as np
import tiktoken
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 400   # tokens per chunk
CHUNK_OVERLAP = 60 # tokens overlap between chunks
TOP_K = 4          # chunks to retrieve per query

_enc = tiktoken.encoding_for_model("gpt-4o-mini")
client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)
    return client


# ── Text chunking ──────────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks by token count."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_enc.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


# ── Embeddings & index ─────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """Return (N, D) float32 array of embeddings."""
    c = get_client()
    response = c.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [item.embedding for item in response.data]
    return np.array(vecs, dtype="float32")


def build_index(chunks: list[str]) -> faiss.IndexFlatIP:
    """Embed chunks and build a FAISS inner-product index (cosine after normalization)."""
    vecs = embed_texts(chunks)
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(query: str, index: faiss.IndexFlatIP, chunks: list[str]) -> list[dict]:
    """Return top-k chunks with their similarity scores."""
    q_vec = embed_texts([query])
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, TOP_K)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({"chunk": chunks[idx], "score": float(score), "idx": int(idx)})
    return results


# ── Generation ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document assistant. Answer the user's question
using ONLY the context provided. If the answer is not in the context, say so clearly.
Keep answers concise and cite the relevant passage when helpful."""


def generate_answer(query: str, retrieved: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]: {r['chunk']}" for i, r in enumerate(retrieved)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    c = get_client()
    resp = c.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    return resp.choices[0].message.content
