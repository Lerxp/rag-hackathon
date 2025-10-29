"""Lightweight BM25/BM25+ querying utilities.

This module loads a prebuilt BM25 index (JSON) and provides helpers to
tokenize queries, score documents using a BM25-style function, and return the
top matches. If the index file is missing, it attempts to build it on the fly
by calling `bm25_index.build_bm25_index()`.

Environment:
- `BM25_INDEX_PATH` (optional): path to the JSON index. Defaults to
  `./data/bm25_index.json`.
"""

import json
import os
import math
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# Location of the serialized BM25 index built by `bm25_index.py`.
# This can be overridden via the BM25_INDEX_PATH environment variable.
INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.json")

def tokenize(text):
    """Tokenize a string into simple alphanumeric, lowercased tokens.

    - Splits on whitespace
    - Keeps only tokens where `str.isalnum()` is True
    - Drops very short tokens (length <= 2)

    This intentionally keeps the logic minimal for speed and portability. For
    production usage, consider a more robust tokenizer (e.g., with stemming,
    stopwords, punctuation handling).
    """
    return [w.lower() for w in text.split() if w.isalnum() and len(w) > 2]

def score_bm25(query_tokens, tf, df, doc_len, avgdl, N, k1=1.5, b=0.75, delta=1.0):
    """Compute BM25/BM25+ scores for a set of documents.

    Parameters
    - query_tokens: list of pre-tokenized query terms
    - tf: dict mapping doc_id -> {term -> term frequency in doc}
    - df: dict mapping term -> document frequency across the corpus
    - doc_len: dict mapping doc_id -> document length (token count)
    - avgdl: average document length across the corpus
    - N: total number of documents in the corpus
    - k1, b: standard BM25 hyperparameters
    - delta: additive term for BM25+ to reduce the length normalization bias

    Returns
    - dict mapping doc_id -> score

    Notes
    - IDF uses log(1 + (N - df + 0.5)/(df + 0.5)) to avoid negative values.
    - The normalization applies BM25+ by using (f + delta) in the numerator.
    """
    scores = {}
    for doc_id in tf:
        score = 0.0
        for term in query_tokens:
            if term in tf[doc_id]:
                f = tf[doc_id][term]
                df_term = df.get(term, 0)
                # Inverse document frequency with standard BM25 smoothing
                idf = math.log(1 + (N - df_term + 0.5) / (df_term + 0.5))
                # Classic BM25 (without +) shown below for reference:
                # norm = f * (k1 + 1) / (f + k1 * (1 - b + b * doc_len[doc_id] / avgdl))
                # BM25+ normalization: use (f + delta) to lessen length bias
                norm = (f + delta) * (k1 + 1) / (f + k1 * (1 - b + b * doc_len[doc_id] / avgdl))
                score += idf * norm
        # Only store non-zero scores to keep result sparse
        if score > 0:
            scores[doc_id] = score
    return scores

def query_bm25(query: str, top_k: int = 5):
    """Query the BM25 index and return top-k matches.

    Parameters
    - query: raw query string (will be tokenized by `tokenize`)
    - top_k: number of results to return

    Behavior
    - If the index JSON at `INDEX_PATH` is missing, attempts to build it by
      calling `bm25_index.build_bm25_index()`.
    - Loads index components and computes BM25 scores for documents.
    - Returns a list of tuples: (text, meta, score), sorted by score desc.
    """
    from pathlib import Path
    import json
    from bm25_index import build_bm25_index  # import the indexer

    # If the BM25 index file doesn't exist, build it automatically
    if not Path(INDEX_PATH).exists():
        print(f"[!] BM25 index not found at {INDEX_PATH}")
        print("[→] Building new BM25 index from Chroma documents...")
        try:
            build_bm25_index()
        except Exception as e:
            print(f"[error] Failed to build BM25 index: {e}")
            return []

    # Load the freshly created (or preexisting) index
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception as e:
        print(f"[error] Failed to read BM25 index: {e}")
        return []

    # Unpack serialized index components
    tf = index["tf"]
    df = index["df"]
    doc_len = index["doc_len"]
    avgdl = index["avgdl"]
    N = index["N"]
    ids = index["ids"]
    texts = index["texts"]
    metas = index["metas"]

    # Score documents and rank by descending score
    query_tokens = tokenize(query)
    scores = score_bm25(query_tokens, tf, df, doc_len, avgdl, N)

    top_hits = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for doc_id, score in top_hits:
        try:
            # Map the string doc_id back to its text and metadata
            # Note: linear search; could be optimized by a reverse map.
            idx = ids.index(doc_id)
            results.append((texts[idx], metas[idx], score))
        except ValueError:
            continue  # skip missing doc_id if index mismatch

    return results
