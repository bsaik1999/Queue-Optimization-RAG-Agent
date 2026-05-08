from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(r"C:\Users\SaiKrishna\OneDrive\Desktop\queuing project")

VECTOR_DB_DIR = BASE_DIR / "vector_db"


def test_retrieval():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embedding_model
    )

    query = "Which Manhattan zones had the biggest driver shortages?"

    results = vectorstore.similarity_search(query, k=3)

    print("\nTop retrieved documents:\n")

    for doc in results:
        print(doc.page_content)
        print("\n-----------------\n")


if __name__ == "__main__":
    test_retrieval()