import os, sys, json, requests, textwrap
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Load config
load_dotenv()
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
embedder = embedding_functions.OllamaEmbeddingFunction(model_name=EMBED_MODEL, url=OLLAMA_URL)
collection = client.get_or_create_collection(name="docs", embedding_function=embedder)

def retrieve(query):
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
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    hits = retrieve(query)

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
        generate_stream(prompt)
    else:
        answer = generate(prompt)
        print(answer)

    print("\n=== TOP MATCHES ===")
    for i, (_, meta, score) in enumerate(hits[:TOP_K], 1):
        print(f"[{i}] {meta['source_file']} p.{meta['page_number']} — score={score:.3f}")

if __name__ == "__main__":
    main()
