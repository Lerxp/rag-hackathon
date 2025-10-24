import os
import json
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from collections import defaultdict
from math import log

# Load config
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.json")

# Tokenization helper
def tokenize(text):
    return [w.lower() for w in text.split() if w.isalnum() and len(w) > 2]

def build_bm25_index():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(name="docs")

    result = col.get(include=["documents", "metadatas"], limit=None)
    documents = result["documents"]
    metadatas = result["metadatas"]
    ids = result["ids"]

    N = len(documents)
    df = defaultdict(int)              # doc frequency
    tf = defaultdict(lambda: defaultdict(int))  # tf[doc_id][token]
    doc_len = {}

    for i, doc in enumerate(documents):
        doc_id = ids[i]
        tokens = tokenize(doc)
        doc_len[doc_id] = len(tokens)

        seen = set()
        for token in tokens:
            tf[doc_id][token] += 1
            if token not in seen:
                df[token] += 1
                seen.add(token)

    avgdl = sum(doc_len.values()) / N

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

    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)

    print(f"[✓] BM25 index saved to {INDEX_PATH} with {N} references.")

if __name__ == "__main__":
    build_bm25_index()
