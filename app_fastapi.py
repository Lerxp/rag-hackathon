"""Minimal FastAPI wrapper around the hybrid RAG pipeline.

Exposes a tiny HTTP API that:
- Serves the static frontend from `static/` at `/`.
- Answers questions via `/ask?query=...` using `answer_hybsrch` (hybrid vector + BM25).

Intended for local-only use; host/port are set to loopback by default.
"""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import answer_hybsrch  # imports retrieve(), build_prompt(), generate()

# Local-only config (binds to loopback by default)
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8000

app = FastAPI(title="Local RAG API", description="Local-only RAG web interface")
# Serve files from the ./static directory under the /static path
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
def home():
    """Serve static frontend root (index.html)."""
    # Returning a path string works with response_class=FileResponse
    return "static/index.html"

@app.get("/ask")
def ask(query: str = Query(..., description="User question")):
    """Answer a user question using hybrid RAG and return JSON."""
    # Retrieve top matches using vector + BM25 hybrid strategy
    hits, _ = answer_hybsrch.retrieve(query)
    if not hits:
        return JSONResponse({"answer": "No results found."})

    # Build grounded prompt and generate an answer (non-streaming)
    prompt = answer_hybsrch.build_prompt(query, hits)
    answer_text, timing = answer_hybsrch.generate(prompt)

    # Summarize matches for the client; include source, page and score
    return {
        "query": query,
        "answer": answer_text,
        "timing": timing,  # optional but useful
        "matches": [
            {
                "source": h[1].get("source_file"),
                "page": h[1].get("page_number"),
                "score": round(h[2], 3),
                "origin": h[3] if len(h) > 3 else "unknown",
            }
            for h in hits
        ],
    }

if __name__ == "__main__":
    # Run a local development server (no autoreload by default)
    import uvicorn
    uvicorn.run("app_fastapi:app", host=LOCAL_HOST, port=LOCAL_PORT, reload=False)
