import os, sys, json, time, requests, textwrap
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# ------------------------------------
# Config
# ------------------------------------
load_dotenv()
CHROMA_DIR      = os.getenv("CHROMA_DIR", "./data/chroma")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL      = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "gemma:2b")
TOP_K           = int(os.getenv("TOP_K", "4"))
NUM_PREDICT     = int(os.getenv("NUM_PREDICT", "350"))
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.2"))
MIN_SCORE       = float(os.getenv("MIN_SCORE", "0.25"))
ANSWER_TIMEOUT  = int(os.getenv("ANSWER_TIMEOUT", "600"))
STREAM_OUTPUT   = os.getenv("STREAM_OUTPUT", "true").lower() == "true"

# ------------------------------------
# Helpers
# ------------------------------------
def approx_tokens_to_words(tokens: int) -> int:
    # ~0.5 words per token (safe average for English text)
    return int(tokens * 0.5)

# ------------------------------------
# Retrieval
# ------------------------------------
def retrieve(query: str, top_k: int = TOP_K):
    """Hybrid retrieval: dense (Chroma) + lexical (BM25)"""
    t0 = time.perf_counter()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedder = embedding_functions.OllamaEmbeddingFunction(model_name=EMBED_MODEL, url=OLLAMA_URL)
    col = client.get_or_create_collection("docs", embedding_function=embedder)
    res = col.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])
    vector_hits = [
        (doc, meta, 1.0 - dist, "vector")    # 🟩 mark vector source
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]

    # BM25 fallback
    try:
        from bm25_query import query_bm25
        bm25_hits_raw = query_bm25(query, top_k=top_k)
        bm25_hits = [(doc, meta, score, "bm25") for doc, meta, score in bm25_hits_raw]  # 🟦 mark BM25 source
    except Exception as e:
        print(f"[warn] BM25 fallback failed: {e}")
        print("Confirm that your index exists. Delete it and rerun bm25_index.py if necessary.")
        bm25_hits = []

    # Combine + dedupe by (source_file, page_number)
    seen, combined = set(), []
    for hit in vector_hits + bm25_hits:
        meta = hit[1]
        key = (meta.get("source_file"), meta.get("page_number"))
        if key not in seen:
            combined.append(hit)
            seen.add(key)

    combined.sort(key=lambda x: x[2], reverse=True)
    t1 = time.perf_counter()
    return combined[:top_k], (t1 - t0)

# ------------------------------------
# Prompt construction
# ------------------------------------
def build_prompt(query, hits):
    # --- Flatten & sanitize hits ---
    cleaned_hits = []
    for h in hits:
        # handle malformed single values
        if not isinstance(h, (list, tuple)):
            continue

        # case: ((text, meta), score, source)
        if len(h) >= 3 and isinstance(h[0], (list, tuple)) and len(h[0]) == 2:
            text, meta = h[0]
            score = h[1]
            source = h[2] if len(h) > 2 else "unknown"
            cleaned_hits.append((text, meta, score, source))

        # case: (text, meta, score, source) — expected
        elif len(h) >= 3 and isinstance(h[2], (int, float)):
            text = h[0]
            meta = h[1]
            score = h[2]
            source = h[3] if len(h) > 3 else "unknown"
            cleaned_hits.append((text, meta, score, source))

    # --- Filter by score ---
    relevant = [h for h in cleaned_hits if h[2] >= MIN_SCORE]
    if not relevant:
        relevant = cleaned_hits[:2]

    # --- Build prompt context ---
    context_blocks, citations = [], []
    for i, (chunk, meta, score, source) in enumerate(relevant, 1):
        src = meta.get("source_file", "?")
        page = meta.get("page_number", "?")
        label = f"{src} p.{page} [{source}]"
        context_blocks.append(f"[{i}] {label}\n{chunk}")
        citations.append(label)

    context = "\n\n".join(context_blocks)
    if len(context) > 8000:
        context = context[:8000] + "\n… [truncated]"

    cite_note = ", ".join(citations)
    approx_words = approx_tokens_to_words(NUM_PREDICT)

    system = textwrap.dedent(f"""
    You are a precise Q&A assistant. Answer ONLY using the CONTEXT.
    - Provide a structured, detailed answer.
    - Be as thorough but as efficient as you can be in {approx_words} words or fewer.
    - If the answer is not fully supported, say "I don't know based on the provided documents."
    - Cite sources inline as (File p.Page [Source]) at the END of each sentence you claim.
    """).strip()

    user = textwrap.dedent(f"""
    QUESTION:
    {query}

    CONTEXT:
    {context}

    INSTRUCTIONS:
    - Include source references like ({cite_note}) after each supported claim.
    - Use content verbatim where appropriate.
    """).strip()

    return f"<SYSTEM>\n{system}\n</SYSTEM>\n<USER>\n{user}\n</USER>"


# ------------------------------------
# Generation
# ------------------------------------
def generate(prompt):
    t0 = time.perf_counter()
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
            "stream": False
        },
        timeout=ANSWER_TIMEOUT
    )
    t1 = time.perf_counter()
    r.raise_for_status()
    data = r.json()

    def ns_to_s(ns): return (float(ns) / 1e9) if ns is not None else None
    timing = {
        "retrieval_s": None,
        "generation_s": (t1 - t0),
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_s": ns_to_s(data.get("eval_duration")),
        "prompt_eval_s": ns_to_s(data.get("prompt_eval_duration")),
        "load_s": ns_to_s(data.get("load_duration")),
    }
    return data.get("response", "").strip(), timing

def generate_stream(prompt):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
        "stream": True
    }

    t0 = time.perf_counter()
    eval_count = prompt_eval_count = None

    with requests.post(url, json=payload, stream=True, timeout=ANSWER_TIMEOUT) as r:
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            # capture incremental output
            if "response" in data:
                print(data["response"], end="", flush=True)
            # capture counters (appear only in final message)
            if "eval_count" in data:
                eval_count = data.get("eval_count")
            if "prompt_eval_count" in data:
                prompt_eval_count = data.get("prompt_eval_count")

    t1 = time.perf_counter()
    print()  # final newline
    return {
        "generation_s": (t1 - t0),
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count
    }

# ------------------------------------
# Main
# ------------------------------------
def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ").strip()

    hits, t_retrieval = retrieve(query)
    if not hits:
        print("No results found.")
        return

    best_score = max(h[2] for h in hits)
    print(f"\nBest match score: {best_score:.3f}")
    if best_score < MIN_SCORE:
        print("⚠️ Low confidence retrieval — answer may be uncertain.\n")

    prompt = build_prompt(query, hits)

    print("\n=== ANSWER ===\n")
    if STREAM_OUTPUT:
        timing = generate_stream(prompt)
    else:
        answer, timing = generate(prompt)
        print(answer)

    print("\n=== TOP MATCHES ===")
    for i, (_, meta, score, source) in enumerate(hits[:TOP_K], 1):
        src_tag = "🔹BM25" if source == "bm25" else "🔸Vector"
        print(f"[{i}] {meta.get('source_file')} p.{meta.get('page_number')} — score={score:.3f} ({src_tag})")

    # concise metrics summary
    total_s = (timing.get("generation_s") or 0) + t_retrieval
    print("\n--- SUMMARY ---")
    print(f"retrieval: {t_retrieval:.2f}s | generation: {timing.get('generation_s', 0):.2f}s | total: {total_s:.2f}s")
    if "prompt_eval_count" in timing:
        p, g = timing.get("prompt_eval_count") or 0, timing.get("eval_count") or 0
        print(f"tokens: {p} prompt + {g} gen = {p + g} total")

if __name__ == "__main__":
    main()
