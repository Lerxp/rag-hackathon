import gradio as gr
import answer_hybsrch  # your hybrid RAG module

def chat_fn(message, history):
    # Step 1: Run retrieval
    hits, _ = answer_hybsrch.retrieve(message)
    if not hits:
        return "No results found in the knowledge base."

    # Step 2: Build LLM prompt
    prompt = answer_hybsrch.build_prompt(message, hits)

    # Step 3: Generate answer (ignore timing tuple)
    answer, _ = answer_hybsrch.generate(prompt)

    # Optional: Show sources under the answer
    sources = "\n".join(
        [f"[{i}] {h[1].get('source_file')} p.{h[1].get('page_number')} ({h[3] if len(h)>3 else '?'})"
         for i, h in enumerate(hits[:answer_hybsrch.TOP_K], 1)]
    )

    return f"{answer}\n\n---\n**Sources:**\n{sources}"

# Gradio chat interface
gr.ChatInterface(
    fn=chat_fn,
    title="Local RAG Chatbot",
    description="Powered by local Ollama + Chroma hybrid retrieval (BM25 + embeddings)."
).launch(server_name="127.0.0.1", share=False)
