"""Simple PDF -> ChromaDB ingestion script.

Reads PDFs from a given folder, chunks pages into overlapping word windows,
embeds them via an Ollama embedding model, and stores them in a persistent
Chroma collection.

Environment variables:
- `CHROMA_DIR`      (default: `./data/chroma`) — persistent DB path
- `EMBED_MODEL`     (default: `nomic-embed-text`) — Ollama embed model name
- `OLLAMA_URL`      (default: `http://localhost:11434`) — Ollama endpoint
- `CHUNK_SIZE`      (default: `400`) — words per chunk
- `CHUNK_OVERLAP`   (default: `100`) — overlapping words between chunks

Usage:
    python ingest.py data/docs
"""

import os, sys, uuid
from pathlib import Path
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load config
load_dotenv()
# Persistent ChromaDB path and embedding configuration (overridable via env)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Chunking controls: window size and overlap measured in words
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

def extract_chunks(pdf: Path):
    """Yield (chunk_text, metadata) pairs for a single PDF.

    Splits each page's text into overlapping word windows using `CHUNK_SIZE`
    and `CHUNK_OVERLAP`. Metadata includes `source_file` and `page_number`.
    """
    with fitz.open(pdf) as doc:
        for page_num, page in enumerate(doc, start=1):
            # Extract raw page text and split into word tokens
            words = page.get_text("text").split()
            # Slide a window with overlap to create dense coverage
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = " ".join(words[i:i + CHUNK_SIZE]).strip()
                if chunk:
                    yield chunk, {"source_file": pdf.name, "page_number": page_num}

def main():
    """CLI entrypoint: ingest all PDFs in the provided folder."""
    if len(sys.argv) < 2:
        print("Usage: python ingest.py data/docs")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    pdfs = list(in_path.glob("*.pdf"))

    if not pdfs:
        print("No PDFs found.")
        return

    # Initialize persistent Chroma client and embedding function via Ollama
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedder = embedding_functions.OllamaEmbeddingFunction(model_name=EMBED_MODEL, url=OLLAMA_URL)
    collection = client.get_or_create_collection("docs", embedding_function=embedder)

    total = 0
    for pdf in pdfs:
        # Extract chunked text and simple metadata per page
        chunks = list(extract_chunks(pdf))
        if not chunks:
            print(f"[skip] {pdf.name} – no text found")
            continue

        # Prepare batched add to avoid very large requests
        docs = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]

        BATCH_SIZE = 8  # Adjust if needed
        for j in range(0, len(docs), BATCH_SIZE):
            collection.add(
                documents=docs[j:j+BATCH_SIZE],
                metadatas=metas[j:j+BATCH_SIZE],
                ids=ids[j:j+BATCH_SIZE],
            )
            print(f"[batch {j//BATCH_SIZE + 1}/{(len(docs)-1)//BATCH_SIZE + 1}] added {len(docs[j:j+BATCH_SIZE])} chunks…")

        print(f"[+] {pdf.name}: {len(docs)} chunks")
        total += len(docs)

    print(f"\nIngest complete. {total} chunks added.")

if __name__ == "__main__":
    main()
