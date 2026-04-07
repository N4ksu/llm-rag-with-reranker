# 🧠 LLM RAG with Reranker

A local AI-powered **Retrieval-Augmented Generation (RAG)** app that lets you upload a PDF and ask questions about it. It uses Ollama for the LLM, ChromaDB for vector storage, and a cross-encoder for smarter answer ranking — all running **100% locally on your computer**.

---

## 📋 Prerequisites

Before you start, make sure you have these installed:

| Tool | Download | Why it's needed |
|------|----------|-----------------|
| **Python 3.11+** | [python.org](https://www.python.org/downloads/) | Runs the app |
| **Git** | [git-scm.com](https://git-scm.com/) | To clone the repo |
| **Ollama** | [ollama.com](https://ollama.com/download) | Runs AI models locally |

---

## 🚀 Step-by-Step Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/N4ksu/llm-rag-with-reranker.git
cd llm-rag-with-reranker
```

### Step 2 — Create a virtual environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> ✅ You'll see `(venv)` at the start of your terminal line when it's active.

### Step 3 — Install Python dependencies

```bash
pip install -r requirements/requirements.txt
```

> ⏳ This may take a few minutes the first time.

### Step 4 — Pull the required Ollama model

Make sure Ollama is running, then pull the LLM:

```bash
ollama pull llama3.2:3b
```

> ⏳ This downloads ~2GB. Wait for it to finish.

### Step 5 — Create the configuration file

Create a file named `.env` in the project root folder with this content:

```
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=mxbai-embed-large:latest
LLM_MODEL=llama3.2:3b
USE_OLLAMA_EMBEDDING=false
```

> 💡 `USE_OLLAMA_EMBEDDING=false` means it uses a fast local model for embeddings. Change to `true` if you want to use Ollama for embeddings too (requires pulling an embedding model).

### Step 6 — Run the app

```bash
# Windows
.\venv\Scripts\python.exe -m streamlit run app.py

# Mac/Linux
python -m streamlit run app.py
```

Open your browser and go to: **http://localhost:8501**

---

## 📖 How to Use

1. **Upload a PDF** — click "Browse files" in the left sidebar
2. **Click ⚡ Process** — wait for the green ✅ success message (first run takes longer)
3. **Type your question** in the text box
4. **Click 🔥 Ask** — the AI will answer based on your document

---

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `ollama` | Runs LLM locally via Ollama |
| `chromadb` | Vector database for storing embeddings |
| `langchain-community` | PDF document loader |
| `langchain-text-splitters` | Splits PDF into text chunks |
| `sentence-transformers` | Fast local embeddings + cross-encoder reranking |
| `PyMuPDF` | Reads and parses PDF files |
| `python-dotenv` | Loads config from `.env` file |

---

## ❗ Common Errors & Fixes

### `Could not find import of chromadb / ollama / streamlit`
Your dependencies aren't installed. Run:
```bash
pip install -r requirements/requirements.txt
```

---

### `PermissionError: [WinError 32] The process cannot access the file`
The PDF temp file is still open. This is fixed in the current version automatically.

---

### `IndexError: list index out of range in upsert`
Usually means the embedding service failed. Check that Ollama is running:
```bash
ollama list
```

---

### `Embedding dimension X does not match collection dimensionality Y`
You switched embedding models after already processing a document. Delete the old database and start fresh:

**Windows:**
```powershell
# Stop the app first (Ctrl+C), then:
Remove-Item -Recurse -Force .\demo-rag-chroma
```

**Mac/Linux:**
```bash
rm -rf demo-rag-chroma/
```

Then restart the app and process your PDF again.

---

### `StreamlitSetPageConfigMustBeFirstCommandError`
This is fixed in the current version. If you see it, make sure you're running the latest `app.py`.

---

### `Error: max retries exceeded / TLS handshake timeout` (when pulling a model)
Network issue when downloading. Try:
1. Run `ollama pull <model-name>` again — it resumes where it left off
2. Try a different network or VPN
3. Try at a different time of day

---

### `❌ Cannot reach Ollama` (shown in the app)
Ollama is not running. Start it:
- **Windows/Mac**: Open the Ollama app from your system tray / Applications
- Or run: `ollama serve` in a terminal

---

### Processing is very slow
- First-time processing is slower because models load into memory
- Subsequent questions are faster (models are cached)
- Make sure `USE_OLLAMA_EMBEDDING=false` in your `.env` for fastest processing

---

## ⚙️ Configuration (`.env` options)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `LLM_MODEL` | `llama3.2:3b` | Model used to generate answers |
| `EMBEDDING_MODEL` | `mxbai-embed-large:latest` | Ollama model for embeddings (only used if `USE_OLLAMA_EMBEDDING=true`) |
| `USE_OLLAMA_EMBEDDING` | `false` | Set to `true` to use Ollama for embeddings |

---

## 📁 Project Structure

```
llm-rag-with-reranker/
├── app.py                    # Main application
├── .env                      # Your config (not committed to git)
├── requirements/
│   ├── requirements.txt      # Python dependencies
│   └── requirements-dev.txt  # Dev tools (ruff linter)
└── demo-rag-chroma/          # Vector database (auto-created, not committed)
```
