# Queue Optimization RAG Agent

A Streamlit-based decision-support project for analyzing NYC ride-matching queues. The project combines queueing theory, driver repositioning recommendations, vector search, and a local LLM through Ollama to answer operational questions about driver shortages.

## Features

- Builds hourly passenger-arrival and driver-activity features from NYC taxi/FHV trip data.
- Estimates queue stability and M/M/1-style wait times.
- Recommends additional driver capacity for shortage zones.
- Generates RAG documents from queue analysis results.
- Builds a Chroma vector store using Hugging Face embeddings.
- Provides a Streamlit UI for RAG questions and what-if queue simulations.

## Project Structure

```text
code/
  Initial.py                # Build queue features from raw trip data
  mm1_wait_time.py          # Estimate wait times and queue status
  Optimization.py           # Recommend driver repositioning
  Document_generator.py     # Create RAG-ready documents
  vector_store.py           # Build the Chroma vector database
  rag_answer.py             # Ask questions using retrieval + Ollama
  Streamlit_api.py          # Streamlit app
  agenttools.py             # Queue-analysis helper functions
  test.py                   # Retrieval smoke test
```

Local folders such as `.venv/`, `data/`, and `vector_db/` are ignored by Git because they can be large or machine-specific.

## Requirements

- Python 3.10+
- Ollama installed locally
- Ollama model `phi3` pulled locally
- NYC taxi/FHV data files placed under `data/raw/`

Install Python packages:

```bash
pip install -r requirements.txt
```

Pull the local LLM model:

```bash
ollama pull phi3
```

## Data Setup

Create this folder structure:

```text
data/
  raw/
    yellow_tripdata_2024-01.parquet
    fhv_tripdata_2024-01.parquet
    taxi_zone_lookup.csv
  processed/
```

The raw NYC taxi and FHV files can be downloaded from the official NYC Taxi & Limousine Commission trip record data site.

## Pipeline

Run the scripts from the project root in this order:

```bash
python code/Initial.py
python code/mm1_wait_time.py
python code/Optimization.py
python code/Document_generator.py
python code/vector_store.py
```

## Run the App

Start the Streamlit app:

```bash
streamlit run code/Streamlit_api.py
```

Then open the local URL shown by Streamlit in your browser.

## Notes

The code currently uses local absolute paths for `BASE_DIR`. If you move the project to another machine, update the `BASE_DIR` values in the Python files or refactor them to use paths relative to the repository root.
