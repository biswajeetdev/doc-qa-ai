# Contributing

Thanks for your interest in contributing to doc-qa-ai!

## Getting started

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-idea`
3. Make your changes and test locally
4. Open a pull request with a clear description

## Local setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY
python app.py
```

## Ideas for contributions

- Support additional file formats (DOCX, TXT, CSV)
- Add a web UI for uploading documents
- Improve chunking strategy for better retrieval accuracy
- Add streaming responses
- Write tests for the RAG pipeline
