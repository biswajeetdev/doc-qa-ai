# doc-qa-ai

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat&logo=meta&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)

RAG-powered document Q&A — upload any PDF and ask questions in plain language. Runs fully free with a Groq key and local embeddings.

## How it works

1. Upload a PDF (up to 10 MB) or paste text
2. Text is chunked and embedded locally using **sentence-transformers** (`all-MiniLM-L6-v2`) — no API call, no cost
3. Embeddings are indexed in FAISS for fast cosine similarity search
4. Your question is embedded the same way and matched against the index
5. Top-4 chunks are passed to **Groq** (Llama 3.3 70B) for a grounded answer with source citations

## Stack

| Layer | Choice |
|---|---|
| LLM | Groq / Llama 3.3 70B (free tier) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local, no API key) |
| Vector store | FAISS (CPU) |
| UI | Streamlit |
| PDF parsing | pypdf |
| Alt provider | OpenAI (switchable via sidebar) |

## Setup

```bash
git clone https://github.com/biswajeetdev/doc-qa-ai
cd doc-qa-ai
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
streamlit run app.py
```

Get a free Groq key (no credit card) at [console.groq.com/keys](https://console.groq.com/keys).

The sentence-transformers model (~22 MB) downloads automatically on first run.

## Security notes

- Chunk text is HTML-escaped before rendering to prevent XSS
- PDF uploads capped at 10 MB
- API key entered in-session only, never stored to disk
