# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Ollama
Download and install from [ollama.ai](https://ollama.ai)

### Step 2: Pull Required Models
```bash
ollama pull nomic-embed-text
ollama pull llama2
```

### Step 3: Start Ollama Service
```bash
ollama serve
```
Keep this terminal open in the background.

### Step 4: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run the System

**Option A: Command-line Interface**
```bash
python quickstart.py
```

**Option B: Web UI (Streamlit)**
```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```

**Option C: Direct Python**
```bash
python rag_system.py
```

## 📝 Adding Your Documents

1. **Create PDF files** or download some
2. **Add to the documents folder:**
   ```bash
   cp your_document.pdf documents/
   ```
3. **Run the system** - it will automatically ingest PDFs

## 💡 Example Queries

- "What are the main topics in this document?"
- "Summarize the key findings"
- "What is [concept] explained in the documents?"
- "List all important dates mentioned"
- "Compare and contrast [topic A] with [topic B]"

## 🎯 Common Issues & Solutions

### Ollama not found
```
Error: Cannot connect to Ollama
```
**Solution:** Make sure `ollama serve` is running in another terminal

### Model not found
```
Error: model llama2 not found
```
**Solution:** Pull the model first:
```bash
ollama pull llama2
```

### Out of memory
```
Error: CUDA out of memory
```
**Solutions:**
- Close other applications
- Use fewer context chunks (reduce `num_results`)
- Use a smaller model (try `neural-chat`)

### Slow responses
**Solutions:**
- Ensure GPU is being used (check Ollama settings)
- Reduce chunk size
- Use a smaller model

## 📚 Project Structure

```
local-rag-qa/
├── rag_system.py           # Core RAG implementation
├── quickstart.py           # Interactive CLI
├── app.py                  # Streamlit web UI
├── setup.py               # Environment setup
├── requirements.txt       # Python dependencies
├── README.md             # Full documentation
├── QUICKSTART.md         # This file
├── documents/            # Your PDF files go here
└── vector_store/         # Auto-generated embeddings
```

## 🔧 Configuration

### Adjust Performance

Edit `rag_system.py`:

```python
# Larger chunks = more context, slower search
chunk_size=1000  # Default: 500

# More results = better coverage, slower
num_results=5    # Default: 3

# Different model
query_model="mistral"  # Instead of "llama2"
```

### Use GPU

Set environment variable before running Ollama:
```bash
# Linux/Mac
export CUDA_VISIBLE_DEVICES=0
ollama serve

# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES=0
ollama serve
```

## 📖 Next Steps

1. **Explore the full README.md** for advanced features
2. **Try different models** - llama3, mistral, neural-chat
3. **Fine-tune parameters** for your use case
4. **Build custom interfaces** using the RAG system

## 🎓 Learn More

- [Ollama Documentation](https://ollama.ai)
- [Chroma DB Guide](https://docs.trychroma.com/)
- [RAG Fundamentals](https://python.langchain.com/docs/use_cases/question_answering/)

## ⚡ Pro Tips

1. **Organize documents**: Use clear filenames
2. **Test with small PDFs first**: Easier to debug
3. **Monitor resources**: Use `ollama ps` to check running models
4. **Batch queries**: Process multiple questions for efficiency
5. **Update models**: Run `ollama pull nomic-embed-text` periodically

---

**Happy questioning! 🚀**
