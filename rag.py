from llm import generate_response
from vector_store import retrieve_context

def jarvis_answer(query):
    context = retrieve_context(query)
    prompt = f"""You are an enterprise AI assistant.
Use the context below to answer accurately.

Context:
{context}

User Question:
{query}

Answer:
"""
    return generate_response(prompt)
