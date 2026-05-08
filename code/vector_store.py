import pandas as pd
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(r"C:\Users\SaiKrishna\OneDrive\Desktop\queuing project")

INPUT_FILE = BASE_DIR / "data" / "processed" / "queue_rag_documents.csv"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

def build_vector_store():
    df = pd.read_csv(INPUT_FILE)
    
    documents = df["document"].dropna().tolist()
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embedding_model,
        persist_directory=str(VECTOR_DB_DIR)
    )

    print("Vector store successfully created.")
    print(f"Saved to: {VECTOR_DB_DIR}")


if __name__ == "__main__":
    build_vector_store()