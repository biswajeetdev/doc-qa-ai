# doc-qa-ai

RAG-powered document Q&A — upload any PDF and ask questions using natural language.

## How it works

1. Upload a PDF
2. Text is chunked and embedded using OpenAI embeddings
3. Embeddings are indexed in FAISS for fast vector search
4. Your question is embedded and matched against the index
5. The top chunks are passed to GPT for a grounded answer

## Stack

- **Language:** Python
- **Embeddings:** OpenAI (`text-embedding-ada-002`)
- **Vector store:** FAISS
- **LLM:** OpenAI GPT
- **API:** FastAPI (app.py)

## Setup

```bash
git clone https://github.com/biswajeetdev/doc-qa-ai
cd doc-qa-ai
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
python app.py
```
