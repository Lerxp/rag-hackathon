from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import answer_hybsrch  # imports retrieve(), build_prompt(), generate()

# Local-only config
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8000

app = FastAPI(title="Local RAG API", description="Local-only RAG web interface")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
def home():
    """Serve static frontend"""
    return "static/index.html"

@app.get("/ask")
def ask(query: str = Query(..., description="User question")):
    """Respond to a question using hybrid RAG"""
    hits, _ = answer_hybsrch.retrieve(query)
    if not hits:
        return JSONResponse({"answer": "No results found."})

    prompt = answer_hybsrch.build_prompt(query, hits)
    answer_text, timing = answer_hybsrch.generate(prompt)

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
    import uvicorn
    uvicorn.run("app_fastapi:app", host=LOCAL_HOST, port=LOCAL_PORT, reload=False)
