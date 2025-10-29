"""RAG answering script using ChromaDB retrieval and an Ollama-served LLM.

Pipeline:
- Embed the user query with the same embedder used during ingestion.
- Retrieve top-k similar chunks from the `docs` collection in ChromaDB.
- Build a grounded prompt that includes citations for each supporting chunk.
- Generate an answer via Ollama's `/api/generate`, optionally streaming tokens.

Environment variables:
- `CHROMA_DIR`    (default: `./data/chroma`) — persistent Chroma path
- `EMBED_MODEL`   (default: `nomic-embed-text`) — embedding model for Ollama
- `OLLAMA_URL`    (default: `http://localhost:11434`) — Ollama base URL
- `LLM_MODEL`     (default: `gemma:2b`) — LLM to query via Ollama
- `TOP_K`         (default: `4`) — number of neighbors to retrieve
- `NUM_PREDICT`   (default: `350`) — max tokens to generate
- `TEMPERATURE`   (default: `0.2`) — sampling temperature
- `MIN_SCORE`     (default: `0.25`) — minimum similarity to consider relevant
- `ANSWER_TIMEOUT`(default: `600`) — request timeout (seconds)
- `STREAM_OUTPUT` (default: `true`) — stream generation output if true
"""

import os, sys, json, requests, textwrap
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Load config
load_dotenv()
# Retrieval and generation configuration (override via environment)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma:2b")
TOP_K = int(os.getenv("TOP_K", "4"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "350"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))
ANSWER_TIMEOUT = int(os.getenv("ANSWER_TIMEOUT", "600"))
STREAM_OUTPUT = os.getenv("STREAM_OUTPUT", "true").lower() == "true"

client = chromadb.PersistentClient(path=CHROMA_DIR)
# Embedding function powered by Ollama (must match ingest-time embeddings)
embedder = embedding_functions.OllamaEmbeddingFunction(model_name=EMBED_MODEL, url=OLLAMA_URL)
# Use the same `docs` collection populated by ingest.py
collection = client.get_or_create_collection(name="docs", embedding_function=embedder)

def retrieve(query):
    """Retrieve top-k similar chunks for a query from ChromaDB.

    Converts cosine distance returned by Chroma into a similarity score using
    `1 - distance` for readability (higher is better).
    Returns a list of tuples: (document_text, metadata, similarity_score).
    """
    res = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    scores = [1 - d for d in dists]
    return list(zip(docs, metas, scores))

def build_prompt(query, hits):
    """Construct a grounded prompt with inline citations.

    - Filters hits below `MIN_SCORE`; if none remain, keeps the top 2 as a
      minimal context.
    - Assembles labeled context blocks and a citation note for the instructions.
    - Truncates overly long context to keep prompts manageable.
    - Wraps sections in simple tags (<SYSTEM>, <USER>) for clarity.
    """
    relevant = [h for h in hits if h[2] >= MIN_SCORE]
    if not relevant:
        relevant = hits[:2]

    context_blocks, citations = [], []
    for i, (chunk, meta, score) in enumerate(relevant, 1):
        src = meta.get("source_file", "?")
        page = meta.get("page_number", "?")
        label = f"{src} p.{page}"
        context_blocks.append(f"[{i}] {label}\n{chunk}")
        citations.append(label)

    context = "\n\n".join(context_blocks)
    if len(context) > 8000:
        context = context[:8000] + "\n… [truncated]"

    cite_note = ", ".join(citations)

    system = textwrap.dedent(f"""
    You are an expert assistant. Answer the user's question using only the CONTEXT provided.
    - Cite your sources like (Filename p.Page) at the end of each sentence you claim.
    - If the answer is not fully supported, say: "I don't know based on the provided documents."
    """).strip()

    user = textwrap.dedent(f"""
    QUESTION:
    {query}

    CONTEXT:
    {context}

    INSTRUCTIONS:
    - Be concise.
    - Include source references like ({cite_note}) after each supported claim.
    - Use content verbatim if appropriate.
    """).strip()

    return f"<SYSTEM>\n{system}\n</SYSTEM>\n<USER>\n{user}\n</USER>"

def generate(prompt):
    """Synchronous generation via Ollama `/api/generate`.

    Returns the final response string (non-streaming).
    Raises for HTTP errors and applies a request timeout.
    """
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": NUM_PREDICT
            },
            "stream": False
        },
        timeout=ANSWER_TIMEOUT
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()

def generate_stream(prompt):
    """Stream tokens from Ollama and print them as they arrive.

    The endpoint emits JSON lines; each may contain a partial "response".
    This function prints the tokens to stdout and ends with a newline.
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
        "stream": True
    }
    with requests.post(url, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    print(data["response"], end="", flush=True)
        print()  # final newline

def main():
    """CLI entrypoint: retrieve context and answer the question."""
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    hits = retrieve(query)

    if not hits:
        print("No results found.")
        return

    # Show the top similarity as a quick confidence indicator
    best_score = max(h[2] for h in hits)
    print(f"\nBest match score: {best_score:.3f}")
    if best_score < MIN_SCORE:
        print("⚠️ Low confidence retrieval — answer may be uncertain.\n")

    prompt = build_prompt(query, hits)

    print("\n=== ANSWER ===\n")
    if STREAM_OUTPUT:
        generate_stream(prompt)
    else:
        answer = generate(prompt)
        print(answer)

    print("\n=== TOP MATCHES ===")
    for i, (_, meta, score) in enumerate(hits[:TOP_K], 1):
        print(f"[{i}] {meta['source_file']} p.{meta['page_number']} — score={score:.3f}")

if __name__ == "__main__":
    main()
