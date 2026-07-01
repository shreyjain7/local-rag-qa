# Local RAG-Based Document Q&A System (Ollama)

A fully local Retrieval-Augmented Generation (RAG) pipeline that ingests PDF documents, embeds them using **nomic-embed-text**, and answers natural language queries via **Llama 3** — no API keys or internet required.

## Features

✨ **100% Local Processing** - All models and data stay on your machine  
📄 **PDF Ingestion** - Automatically extract and process PDF documents  
🔍 **Semantic Search** - Find relevant content using embeddings  
🤖 **LLM-Powered Answers** - Generate contextual answers with Llama 3  
💾 **Persistent Storage** - Vector embeddings stored locally in Chroma DB  
⚡ **Efficient Chunking** - Smart text splitting with overlap for context preservation  

## Prerequisites

- **Ollama** installed and running ([Download](https://ollama.ai))
- Python 3.8+
- At least 8GB RAM (16GB recommended)

## Installation

### 1. Install Ollama
Download and install Ollama from [ollama.ai](https://ollama.ai)

### 2. Pull Required Models
```bash
ollama pull nomic-embed-text  # For embeddings
ollama pull llama2            # For Q&A (or llama3 if available)
```

### 3. Clone and Setup Project
```bash
cd local-rag-qa
python setup.py
```

Or manually:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Add your documents:**
   ```bash
   # Copy your PDF files to the documents/ folder
   cp your_documents.pdf documents/
   ```

2. **Run the system:**
   ```bash
   python rag_system.py
   ```

3. **Ask questions:**
   ```
   Ask a question: What are the main topics covered in this document?
   ```

### Programmatic Usage

```python
from rag_system import LocalRAGSystem

# Initialize
rag = LocalRAGSystem(
    vector_store_path="vector_store",
    embedding_model="nomic-embed-text",
    query_model="llama2"
)

# Ingest PDFs
rag.ingest_pdf("documents/my_document.pdf", "My Document")

# Ask questions
answer = rag.answer_query("What is the main topic?")
print(answer)
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────┐
│         PDF Document Input                      │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│    PDF Text Extraction & Chunking               │
│    (PyPDF, LangChain)                           │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│    Embedding Generation                         │
│    (nomic-embed-text via Ollama)                │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│    Vector Store                                 │
│    (Chroma DB - Persistent Local Storage)       │
└──────────────────┬──────────────────────────────┘
                   ↓
    ┌─────────────────────────────────┐
    │ User Query                      │
    └──────────────┬──────────────────┘
                   ↓
    ┌─────────────────────────────────┐
    │ Embed Query                     │
    │ (nomic-embed-text)              │
    └──────────────┬──────────────────┘
                   ↓
    ┌─────────────────────────────────┐
    │ Semantic Search                 │
    │ (Vector Similarity)             │
    └──────────────┬──────────────────┘
                   ↓
    ┌─────────────────────────────────┐
    │ Retrieve Top Chunks             │
    └──────────────┬──────────────────┘
                   ↓
    ┌─────────────────────────────────────────────┐
    │ Generate Answer                             │
    │ (Llama 3 with Retrieved Context)            │
    └──────────────┬────────────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────────┐
    │ Answer Output to User                       │
    └─────────────────────────────────────────────┘
```

## Configuration

Edit `rag_system.py` to customize:

```python
rag_system = LocalRAGSystem(
    vector_store_path="vector_store",      # Location for embeddings
    embedding_model="nomic-embed-text",    # Embedding model
    query_model="llama2"                   # Query/answer model
)
```

### Chunking Parameters
```python
chunks = rag_system.chunk_text(
    text,
    chunk_size=500,      # Size of each chunk
    chunk_overlap=50     # Overlap between chunks
)
```

### Retrieval Parameters
```python
chunks = rag_system.retrieve_relevant_chunks(
    query,
    num_results=3        # Number of chunks to retrieve
)
```

## Advanced Features

### 1. Batch PDF Loading
```python
rag_system.load_documents_from_folder("documents/")
```

### 2. Collection Statistics
```python
stats = rag_system.get_collection_stats()
print(f"Total chunks: {stats['total_chunks']}")
```

### 3. Custom Retrieval
```python
relevant_chunks = rag_system.retrieve_relevant_chunks(
    query="your question",
    num_results=5
)
for chunk in relevant_chunks:
    print(f"Source: {chunk['source']}")
    print(f"Content: {chunk['content']}")
    print(f"Relevance: {1 - chunk['distance']}")
```

## Performance Tips

1. **Adjust Chunk Size:** Larger chunks = more context, slower search
   ```python
   chunks = rag_system.chunk_text(text, chunk_size=1000)
   ```

2. **Batch Processing:** Use folder loading for multiple documents
   ```python
   rag_system.load_documents_from_folder("documents/")
   ```

3. **Use GPU:** Enable GPU acceleration in Ollama
   ```bash
   # Set environment variable before running Ollama
   CUDA_VISIBLE_DEVICES=0 ollama serve
   ```

## Troubleshooting

### Q: "Connection refused" error
**A:** Make sure Ollama is running
```bash
ollama serve  # In a separate terminal
```

### Q: Models not found
**A:** Pull the required models
```bash
ollama pull nomic-embed-text
ollama pull llama2
```

### Q: Slow performance
**A:** 
- Reduce chunk size for faster retrieval
- Use fewer retrieval results
- Ensure Ollama is using GPU

### Q: Out of memory
**A:**
- Reduce `num_results` in retrieval
- Use smaller chunk sizes
- Close other applications

## Project Structure

```
local-rag-qa/
├── rag_system.py          # Main RAG system class
├── setup.py               # Setup and verification script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── documents/            # Directory for PDF documents
│   └── sample.txt       # Sample document
└── vector_store/        # Persistent vector embeddings
```

## Dependencies

- **ollama** - Python client for Ollama
- **chromadb** - Vector database for embeddings
- **pypdf** - PDF text extraction
- **langchain** - Text splitting utilities
- **python-dotenv** - Environment configuration

## Alternative Models

You can use different Ollama models:

### For Embeddings:
```bash
ollama pull nomic-embed-text  # Recommended (nomic-ai/nomic-embed-text)
```

### For Q&A:
```bash
ollama pull llama2         # 7B, fast
ollama pull llama3         # Newer, better quality
ollama pull mistral        # 7B, efficient
ollama pull neural-chat    # 7B, optimized
```

## Limitations

- Performance depends on your hardware
- Response quality varies with document relevance and model size
- All processing is local, so responses may be slower than cloud APIs
- Requires adequate disk space for vector embeddings

## Future Enhancements

- [ ] Web UI (Streamlit/Flask)
- [ ] Support for more document formats (DOCX, TXT, MD)
- [ ] Multi-language support
- [ ] Document summarization
- [ ] Memory/conversation history
- [ ] Citation tracking
- [ ] Fine-tuned models for specific domains

## License

MIT License - Feel free to use and modify for your needs.

## Contributing

Contributions welcome! Please feel free to submit PRs or report issues.

## Resources

- [Ollama Documentation](https://ollama.ai)
- [Chroma DB](https://www.trychroma.com/)
- [LangChain](https://python.langchain.com/)
- [Nomic Embed](https://www.nomic.ai/)

---

**Built with ❤️ for local, private document Q&A**
