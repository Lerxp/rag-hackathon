"""Simple ChromaDB query helper.

Loads a persistent Chroma collection, embeds the input query via an Ollama
embedding model, searches top-k similar chunks, and prints results above a
minimum similarity threshold.

Environment variables:
- `CHROMA_DIR`  (default: `./data/chroma`) – path for persistent DB
- `EMBED_MODEL` (default: `nomic-embed-text`) – Ollama embedding model
- `OLLAMA_URL`  (default: `http://localhost:11434`) – Ollama endpoint
- `TOP_K`       (default: `4`) – number of results to retrieve
- `MIN_SCORE`   (default: `0.25`) – minimum similarity to display
"""

import os, sys
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
# Configuration (overridable via environment)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))

client = chromadb.PersistentClient(path=CHROMA_DIR)
# Embedding function backed by a local/remote Ollama server
embedder = embedding_functions.OllamaEmbeddingFunction(model_name=EMBED_MODEL, url=OLLAMA_URL)
# Use or create the `docs` collection where `ingest.py` added chunks
collection = client.get_or_create_collection(name="docs", embedding_function=embedder)

def ask(query):
    """Query the `docs` collection and print top matches.

    - Uses cosine distance returned by Chroma and converts to similarity as
      `1 - distance` for easier interpretation (higher is more similar).
    - Filters out results with similarity below `MIN_SCORE`.
    """
    res = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    # Chroma returns lists-of-lists (one per query); we use the first
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]
    # Convert cosine distance (0 = identical, 2 = opposite for unnormalized)
    # to a simple similarity score in [0, 1] using 1 - d
    scores = [1 - d for d in distances]

    for i, (text, meta, score) in enumerate(zip(docs, metas, scores), start=1):
        if score < MIN_SCORE:
            continue
        print(f"\n[{i}] score={score:.3f} | {meta['source_file']} p.{meta['page_number']}")
        # Print a preview of the chunk, flattening newlines
        print("     " + text[:300].replace("\n", " ") + ("…" if len(text) > 300 else ""))

if __name__ == "__main__":
    # Accept query either as CLI args or interactive input
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your question: ")
    ask(query)
