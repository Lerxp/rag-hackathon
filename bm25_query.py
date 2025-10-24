import json
import os
import math
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.json")

# Basic tokenizer
def tokenize(text):
    return [w.lower() for w in text.split() if w.isalnum() and len(w) > 2]

# BM25 scoring function
def score_bm25(query_tokens, tf, df, doc_len, avgdl, N, k1=1.5, b=0.75, delta=1.0):
    scores = {}
    for doc_id in tf:
        score = 0.0
        for term in query_tokens:
            if term in tf[doc_id]:
                f = tf[doc_id][term]
                df_term = df.get(term, 0)
                idf = math.log(1 + (N - df_term + 0.5) / (df_term + 0.5))
#                norm = f * (k1 + 1) / (f + k1 * (1 - b + b * doc_len[doc_id] / avgdl))
                norm = (f + delta) * (k1 + 1) / (f + k1 * (1 - b + b * doc_len[doc_id] / avgdl))
                score += idf * norm
        if score > 0:
            scores[doc_id] = score
    return scores

def query_bm25(query: str, top_k: int = 5):
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

    tf = index["tf"]
    df = index["df"]
    doc_len = index["doc_len"]
    avgdl = index["avgdl"]
    N = index["N"]
    ids = index["ids"]
    texts = index["texts"]
    metas = index["metas"]

    query_tokens = tokenize(query)
    scores = score_bm25(query_tokens, tf, df, doc_len, avgdl, N)

    top_hits = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for doc_id, score in top_hits:
        try:
            idx = ids.index(doc_id)
            results.append((texts[idx], metas[idx], score))
        except ValueError:
            continue  # skip missing doc_id if index mismatch

    return results
