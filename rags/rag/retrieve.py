import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

FAISS_STORE_PATH = Path(__file__).parent.parent / "faiss_store"
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

def search_collection(name: str, query_embedding: np.ndarray, n_results: int) -> list:
    index_file = FAISS_STORE_PATH / f"{name}.index"
    mapping_file = FAISS_STORE_PATH / f"{name}.json"
    
    if not index_file.exists() or not mapping_file.exists():
        return []
    
    index = faiss.read_index(str(index_file))
    with open(mapping_file, "r") as f:
        mapping = json.load(f)
        
    k = min(n_results, index.ntotal)
    if k <= 0:
        return []
        
    # FAISS expects 2D float32 numpy array for querying
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(mapping):
            results.append(mapping[idx]["metadata"])
    return results

def retrieve_evidence(anomaly: str, n_results: int = 2) -> dict:
    # Nomic v1.5 requires prefixing queries with 'search_query: '
    query_embedding = np.array([model.encode(f"search_query: {anomaly}")], dtype=np.float32)
    
    incidents = search_collection("incidents", query_embedding, n_results)
    sops = search_collection("sops", query_embedding, n_results)
    logs = search_collection("maintenance_logs", query_embedding, n_results)
    
    return {
        "incidents": incidents,
        "sops": sops,
        "maintenance_logs": logs
    }
