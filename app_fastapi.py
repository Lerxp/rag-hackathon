from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import answer  # imports retrieve(), build_prompt(), generate()

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
    """Use answer.py functions to respond to a question"""
    hits = answer.retrieve(query)
    if not hits:
        return JSONResponse({"answer": "No results found."})
    prompt = answer.build_prompt(query, hits)
    answer_text = answer.generate(prompt)
    return {
        "query": query,
        "answer": answer_text,
        "matches": [
            {"source": h[1]["source_file"], "page": h[1]["page_number"], "score": round(h[2], 3)}
            for h in hits
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_fastapi:app", host=LOCAL_HOST, port=LOCAL_PORT, reload=False)
