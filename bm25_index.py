"""Build a lightweight BM25/BM25+ compatible index from ChromaDB.

This script reads all documents from the `docs` collection in a persistent
ChromaDB store and constructs a simple JSON index with token statistics used by
BM25-style retrieval (term frequency, document frequency, document lengths,
average document length, and lookups for texts/metadatas/ids).

Notes
- Assumes documents were previously ingested into the `docs` collection
  (e.g., via `ingest.py`).
- The output JSON is consumed by `bm25_query.py` for scoring queries.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from collections import defaultdict

# Load configuration from environment (with defaults)
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.json")

# Tokenization helper
def tokenize(text):
    """Simple, fast tokenizer used for indexing.

    - Lowercases tokens
    - Keeps only alphanumeric tokens
    - Drops tokens of length <= 2

    This mirrors the tokenizer used at query time for consistency.
    """
    return [w.lower() for w in text.split() if w.isalnum() and len(w) > 2]

def build_bm25_index():
    """Build and persist BM25 statistics from the `docs` collection.

    Constructs:
    - tf: term frequencies per doc_id
    - df: document frequencies per token
    - doc_len: tokenized length per document
    - avgdl: average document length
    - ids, texts, metas: passthrough lists from Chroma to enable lookups
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(name="docs")

    # Retrieve all stored chunks and metadata
    result = col.get(include=["documents", "metadatas"], limit=None)
    documents = result["documents"]
    metadatas = result["metadatas"]
    ids = result["ids"]

    N = len(documents)
    # Document frequency: number of docs containing a token at least once
    df = defaultdict(int)
    # Term frequency per doc: tf[doc_id][token] = count in that doc
    tf = defaultdict(lambda: defaultdict(int))
    # Document lengths: number of tokens per document
    doc_len = {}

    for i, doc in enumerate(documents):
        doc_id = ids[i]
        tokens = tokenize(doc)
        doc_len[doc_id] = len(tokens)

        # Use a set to count each token once for DF
        seen = set()
        for token in tokens:
            tf[doc_id][token] += 1
            if token not in seen:
                df[token] += 1
                seen.add(token)

    # Average document length across the corpus
    avgdl = sum(doc_len.values()) / N

    # Package index data for downstream BM25 scoring
    index = {
        "avgdl": avgdl,
        "N": N,
        "doc_len": doc_len,
        "tf": tf,
        "df": df,
        "ids": ids,
        "texts": documents,
        "metas": metadatas,
    }

    # Ensure destination exists and persist as JSON
    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)

    print(f"[✓] BM25 index saved to {INDEX_PATH} with {N} references.")

if __name__ == "__main__":
    build_bm25_index()
