import os, sys, uuid
from pathlib import Path
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load config
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

def extract_chunks(pdf: Path):
    with fitz.open(pdf) as doc:
        for page_num, page in enumerate(doc, start=1):
            words = page.get_text("text").split()
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = " ".join(words[i:i + CHUNK_SIZE]).strip()
                if chunk:
                    yield chunk, {"source_file": pdf.name, "page_number": page_num}

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py data/docs")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    pdfs = list(in_path.glob("*.pdf"))

    if not pdfs:
        print("No PDFs found.")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedder = embedding_functions.OllamaEmbeddingFunction(model_name=EMBED_MODEL, url=OLLAMA_URL)
    collection = client.get_or_create_collection("docs", embedding_function=embedder)

    total = 0
    for pdf in pdfs:
        chunks = list(extract_chunks(pdf))
        if not chunks:
            print(f"[skip] {pdf.name} – no text found")
            continue

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
