import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

FAISS_STORE_PATH = Path(__file__).parent.parent / "faiss_store"

def load_json(filename):
    with open(Path(__file__).parent / "data" / filename) as f:
        return json.load(f)

def ingest_collection(name, documents, embeddings_texts, metadatas, ids, model):
    # Ensure store directory exists
    FAISS_STORE_PATH.mkdir(parents=True, exist_ok=True)

    # Encode embeddings. Nomic v1.5 requires prefixing documents with 'search_document: '
    prefixed_texts = [f"search_document: {text}" for text in embeddings_texts]
    embeddings = np.array(model.encode(prefixed_texts), dtype=np.float32)
    dimension = embeddings.shape[1]

    # Create and populate FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index
    index_file = FAISS_STORE_PATH / f"{name}.index"
    faiss.write_index(index, str(index_file))

    # Save documents/metadata mapping
    mapping = []
    for idx in range(len(ids)):
        mapping.append({
            "id": ids[idx],
            "document": documents[idx],
            "metadata": metadatas[idx]
        })
    
    mapping_file = FAISS_STORE_PATH / f"{name}.json"
    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=2)

def ingest_all():
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    # Incidents
    incidents = load_json("incidents.json")
    ingest_collection(
        name="incidents",
        documents=[f"Machine {i['machine']} issue: {i['issue']} on {i['date']}. Resolution: {i['resolution']}" for i in incidents],
        embeddings_texts=[f"Machine {i['machine']} issue: {i['issue']}" for i in incidents],
        metadatas=incidents,
        ids=[i["id"] for i in incidents],
        model=model
    )

    # SOPs
    sops = load_json("sops.json")
    ingest_collection(
        name="sops",
        documents=[s["rule"] for s in sops],
        embeddings_texts=[s["trigger"] + " " + s["rule"] for s in sops],
        metadatas=sops,
        ids=[s["id"] for s in sops],
        model=model
    )

    # Maintenance logs
    logs = load_json("maintenance_logs.json")
    ingest_collection(
        name="maintenance_logs",
        documents=[f"Machine {l['machine']} on {l['date']}: {l['action']} by {l['technician']}" for l in logs],
        embeddings_texts=[f"Machine {l['machine']} {l['action']}" for l in logs],
        metadatas=logs,
        ids=[l["id"] for l in logs],
        model=model
    )

    print("Ingestion complete (using FAISS).")

if __name__ == "__main__":
    ingest_all()
