import pandas as pd
from pathlib import Path 

BASE_DIR = Path(r"C:\Users\SaiKrishna\OneDrive\Desktop\queuing project")

INPUT_FILE = BASE_DIR / "data" / "processed" / "driver_repositioning_recommendations_jan2024.csv"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "queue_rag_documents.csv"


def create_documents():
    
    df = pd.read_csv(INPUT_FILE)
    
    documents = []
    
    for _ , row in df.iterrows():
    
        document_text = f"""On {row['pickup_hour']} in {row['Zone']}, {row['Borough']},
            passenger arrivals were {int(row['passenger_arrivals'])}
            while driver activity was {int(row['driver_activity'])}.

            The system status was {row['queue_status']}.

           Estimated wait time was {row['estimated_wait_time']}.

            The model recommends adding approximately
            {int(row['extra_drivers_needed'])} additional drivers
            to stabilize the queue.
            """
            
        documents.append(document_text.strip()) 
        
        rag_df = pd.DataFrame({"document": documents})
        
        rag_df.to_csv(OUTPUT_FILE , index = False)
        
        print(f"Saved RAG documents to: {OUTPUT_FILE}")
        
        print(rag_df.head())
        
if __name__ == "__main__":
    create_documents()       