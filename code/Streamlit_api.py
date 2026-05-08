import math
from pathlib import Path

import ollama
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(r"C:\Users\SaiKrishna\OneDrive\Desktop\queuing project")
VECTOR_DB_DIR = BASE_DIR / "vector_db"


def estimate_wait_time(passenger_rate, driver_rate):
    if driver_rate <= passenger_rate:
        return float("inf")
    return passenger_rate / (driver_rate * (driver_rate - passenger_rate))


def estimate_queue_status(passenger_rate, driver_rate):
    if driver_rate == 0 and passenger_rate > 0:
        return "No driver supply"
    elif driver_rate <= passenger_rate:
        return "Unstable / shortage"
    return "Stable"


def calculate_extra_drivers_needed(passenger_rate, driver_rate, safety_buffer=1):
    if driver_rate > passenger_rate:
        return 0
    return max(0, (passenger_rate + safety_buffer) - driver_rate)


def simulate_driver_increase(passenger_rate, driver_rate, percent_increase):
    new_driver_rate = driver_rate * (1 + percent_increase / 100)

    return {
        "passenger_rate": passenger_rate,
        "original_driver_rate": driver_rate,
        "new_driver_rate": round(new_driver_rate, 2),
        "old_status": estimate_queue_status(passenger_rate, driver_rate),
        "new_status": estimate_queue_status(passenger_rate, new_driver_rate),
        "old_wait_time": estimate_wait_time(passenger_rate, driver_rate),
        "new_wait_time": estimate_wait_time(passenger_rate, new_driver_rate),
        "extra_drivers_needed_after": math.ceil(
            calculate_extra_drivers_needed(passenger_rate, new_driver_rate)
        ),
    }


@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embedding_model
    )


def retrieve_context(query, k=5):
    vectorstore = load_vectorstore()
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
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


st.set_page_config(
    page_title="Queue Optimization RAG Agent",
    layout="wide"
)

st.title("🚕 Queue Optimization RAG Agent")
st.write(
    "A GenAI decision-support system for NYC ride-matching queues using RAG, "
    "queueing theory, and driver repositioning optimization."
)

tab1, tab2 = st.tabs(["Ask the RAG Agent", "What-if Queue Simulation"])


with tab1:
    st.subheader("Ask a question about driver shortages")

    query = st.text_input(
        "Question",
        value="Which Manhattan zones had the biggest driver shortages?"
    )

    k = st.slider("Number of retrieved evidence documents", 1, 10, 5)

    if st.button("Ask Agent"):
        with st.spinner("Retrieving evidence and generating answer..."):
            contexts = retrieve_context(query, k=k)
            answer = generate_llm_answer(query, contexts)

        st.markdown("### LLM Answer")
        st.write(answer)

        st.markdown("### Retrieved Evidence")
        for i, context in enumerate(contexts, start=1):
            with st.expander(f"Evidence {i}"):
                st.write(context)


with tab2:
    st.subheader("Run queue what-if simulation")

    col1, col2, col3 = st.columns(3)

    with col1:
        passenger_rate = st.number_input(
            "Passenger arrivals",
            min_value=0,
            value=726
        )

    with col2:
        driver_rate = st.number_input(
            "Driver activity",
            min_value=0,
            value=4
        )

    with col3:
        percent_increase = st.number_input(
            "Driver increase (%)",
            min_value=0,
            value=20
        )

    if st.button("Run Simulation"):
        result = simulate_driver_increase(
            passenger_rate,
            driver_rate,
            percent_increase
        )

        st.markdown("### Simulation Result")

        st.metric("Original Status", result["old_status"])
        st.metric("New Status", result["new_status"])
        st.metric("New Driver Activity", result["new_driver_rate"])
        st.metric(
            "Extra Drivers Still Needed",
            result["extra_drivers_needed_after"]
        )

        st.write("Full result:")
        st.json(result)