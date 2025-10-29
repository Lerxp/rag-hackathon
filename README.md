# Local RAG Hybrid Search Tutorial 

This repository demonstrates a local **Retrieval-Augmented Generation (RAG)** pipeline using **Ollama**, **ChromaDB**, and **BM25+ hybrid search**, complete with both command-line and web interfaces.

All processing happens locally — no cloud APIs or internet access required.

---

## 📂 Project Structure

```
rag-starter/
├── .env                   # Environment and model configuration
├── data/
│   ├── chroma/            # Persistent vector store (auto-created)
│   ├── bm25_index.json    # Lexical index (BM25+) from Chroma docs
│   └── docs/              # Place your PDFs here
│
├── static/
│   ├── index.html        # Static demo frontend for FastAPI demo (app_fastapi.py)
│   ├── style.css         # Styling for static frontend
│   └── script.js         # Frontend logic (JSON or streamed text handling)
│
├── ingest.py              # Chunk + embed documents into ChromaDB
├── bm25_index.py          # Build BM25+ lexical index
├── bm25_query.py          # Query BM25+ index (auto-builds if missing)
│
├── query.py               # Simple retrieval test script
├── answer.py              # Baseline RAG (vector-only)
├── answer_hybsrch.py      # Hybrid RAG (vector + BM25), streaming, timing
│
├── app_fastapi.py         # REST API with FastAPI (JSON responses)
└── app_gradio.py          # Local chat interface using Gradio

```

---

## ⚙️ Configuration (`.env`)

Example:
```
CHROMA_DIR=./data/chroma
EMBED_MODEL=nomic-embed-text
OLLAMA_URL=http://127.0.0.1:11434
LLM_MODEL=gemma:2b
CHUNK_SIZE=400
CHUNK_OVERLAP=100
TOP_K=4
MIN_SCORE=0.25
NUM_PREDICT=350
TEMPERATURE=0.2
ANSWER_TIMEOUT=600
STREAM_OUTPUT=true
```

These control model endpoints, search scope, generation settings, and timing behavior.

---

## 🧩 Python Components

### 🪶 1. Ingest PDFs → Chroma Vector DB (`ingest.py`)
- Reads PDFs under `data/docs/`
- Splits text into overlapping chunks
- Embeds chunks via Ollama (`nomic-embed-text`)
- Stores vectors in a persistent Chroma collection

**Run:**
```bash
python ingest.py data/docs
```

---

### 🧮 2. Build Lexical Index (`bm25_index.py`)
- Scans Chroma documents and computes term frequencies  
- Creates `data/bm25_index.json` for BM25+ retrieval  
- Displays token and document counts when saved  

**Run:**
```bash
python bm25_index.py
```

---

### 🔍 3. Query Lexical Index (`bm25_query.py`)
- Loads or automatically builds BM25 index if missing  
- Uses BM25+ scoring to return keyword-matched chunks  
- Integrates cleanly with `answer_hybsrch.py`  

**Run:**
```bash
python -m bm25_query "What is the dress code policy?"
```

---

### 🧠 4. Vector-Only RAG (`answer.py`)
- Pulls top-K relevant chunks from Chroma  
- Builds a structured prompt and calls Ollama  
- Prints generated answer and top matches  

**Run:**
```bash
python answer.py "Explain the company's remote work policy."
```

---

### ⚡ 5. Hybrid RAG (`answer_hybsrch.py`)
Combines **vector** and **BM25** retrievals:
- Deduplicates overlapping hits  
- Annotates sources (🔸Vector / 🔹BM25)  
- Displays concise timing + token usage summary  

Example:
```
Best match score: 0.792

=== ANSWER ===
<streamed model output>

=== TOP MATCHES ===
[1] EmployeeHandbook.pdf p.7 — score=0.792 (🔸Vector)
[2] WorkplacePolicies.pdf p.4 — score=0.761 (🔹BM25)

--- SUMMARY ---
retrieval: 0.82s | generation: 6.47s | total: 7.29s
tokens: 148 prompt + 372 gen = 520 total
```

**Run:**
```bash
python answer_hybsrch.py "Describe the process for requesting paid leave."
```

---

## 🌐 Web Interfaces

### ⚙️ FastAPI Server (`app_fastapi.py`)
- Serves the static demo at `/` (loads `/static/style.css` and `/static/script.js`).
- Answers questions via `/ask?query=...` using the hybrid pipeline.
- Response mode is controlled by `.env` `STREAM_OUTPUT`:
  - `false` → returns JSON `{ answer, timing, matches }`.
  - `true`  → streams plain text tokens; matches are provided in `X-Matches` header.

**Start:**
```bash
python app_fastapi.py
```
Visit:
```
http://127.0.0.1:8000/ask?query=How+to+report+a+workplace+incident
```

Or open the static frontend:
```
http://127.0.0.1:8000/
```
The page script (`static/script.js`) automatically adapts to the API response:
- If `application/json`, it renders the full answer and matches.
- If `text/plain` streaming, it shows matches from the `X-Matches` header and appends streamed tokens live.

---

### 💬 Gradio Interface (`app_gradio.py`)
- Launches a chat-style browser UI  
- Displays live question + streamed answer  
- Uses same hybrid backend  

**Run:**
```bash
python app_gradio.py
```
Default:
```
http://127.0.0.1:7860/
```

---

## ⚙️ Setup Instructions

### 1. Create & Activate a Virtual Environment
**Windows (PowerShell):**
```powershell
cd C:\LLM\rag-starter
python -m venv .rag-venv
.\.rag-venv\Scripts\Activate.ps1
```

If needed:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS / Linux:**
```bash
cd ~/Projects/rag-starter
python3 -m venv .rag-venv
source .rag-venv/bin/activate
```

---

### 2. Install Dependencies
```bash
pip install chromadb pymupdf python-dotenv requests fastapi uvicorn gradio ollama
```

---

### 3. Start Ollama
```bash
ollama pull gemma:2b
ollama pull nomic-embed-text
ollama serve
```

---

### 4. Ingest → Index → Query
```bash
python ingest.py data/docs
python bm25_index.py
python answer_hybsrch.py "Explain attendance policy."
```

---

## 🧩 Architectural Overview

Below is a conceptual flow of how **Hybrid Retrieval-Augmented Generation** operates in this project:

```
                  ┌─────────────────────────────┐
                  │         User Query          │
                  └──────────────┬──────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           │                                           │
   ┌───────▼────────┐                         ┌────────▼────────┐
   │  Chroma Vector  │                         │   BM25+ Lexical │
   │   Retrieval     │                         │   Retrieval     │
   └───────┬────────┘                         └────────┬────────┘
           │                                           │
           └──────────────┬──────────────┬─────────────┘
                          │   Deduplicate + Merge
                          ▼
               ┌────────────────────────────┐
               │     Combined Context       │
               │  (Top-K Chunks + Metadata) │
               └────────────┬──────────────┘
                            │
                     Prompt Construction
                            │
                            ▼
                ┌─────────────────────────┐
                │  Local LLM via Ollama   │
                │   (e.g., Gemma-2B)      │
                └────────────┬────────────┘
                             │
                      Streaming Output
                             │
                             ▼
                  ┌────────────────────────┐
                  │  Final Answer + Stats  │
                  └────────────────────────┘
```
