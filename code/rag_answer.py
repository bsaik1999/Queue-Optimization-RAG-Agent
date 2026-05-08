from pathlib import Path 

import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(r"C:\Users\SaiKrishna\OneDrive\Desktop\queuing project")
VECTOR_DB_DIR = BASE_DIR / "vector_db"

def retrieve_context(query , k=5):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embedding_model
    )

    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


def generate_llm_answer(query, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are an operations research and queueing analytics assistant.

Use ONLY the retrieved evidence below to answer the user's question.
Be concise, analytical, and decision-oriented.

User question:
{query}

Retrieved evidence:
{context_text}

Answer:
"""

    response = ollama.chat(
        model="phi3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


def answer_question(query):
    contexts = retrieve_context(query, k=5)

    print("\nQuestion:")
    print(query)

    print("\nRetrieved Evidence:")
    for i, context in enumerate(contexts, start=1):
        print(f"\nEvidence {i}:")
        print(context)

    final_answer = generate_llm_answer(query, contexts)

    print("\nLLM Answer:")
    print(final_answer)


if __name__ == "__main__":
    question = "Which Manhattan zones had the biggest driver shortages?"
    answer_question(question)
