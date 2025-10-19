import os, sys
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))

client = chromadb.PersistentClient(path=CHROMA_DIR)
embedder = embedding_functions.OllamaEmbeddingFunction(model_name=EMBED_MODEL, url=OLLAMA_URL)
collection = client.get_or_create_collection(name="docs", embedding_function=embedder)

def ask(query):
    res = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]
    scores = [1 - d for d in distances]  # Convert cosine distance to similarity

    for i, (text, meta, score) in enumerate(zip(docs, metas, scores), start=1):
        if score < MIN_SCORE:
            continue
        print(f"\n[{i}] score={score:.3f} | {meta['source_file']} p.{meta['page_number']}")
        print("     " + text[:300].replace("\n", " ") + ("…" if len(text) > 300 else ""))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your question: ")
    ask(query)
