import gradio as gr
import answer  # your existing module

def chat_fn(message, history):
    hits = answer.retrieve(message)

    if not hits:
        return "No results found in the knowledge base."

    prompt = answer.build_prompt(message, hits)
    reply = answer.generate(prompt)
    return reply

gr.ChatInterface(
    fn=chat_fn,
    title="Local RAG Chatbot",
    description="Powered by your local Ollama + Chroma setup."
).launch(server_name="127.0.0.1", share=False)
