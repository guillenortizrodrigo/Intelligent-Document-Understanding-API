import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

#paths
INDEX_PATH = "vector_index.faiss"
META_PATH = "metadata.pkl"

#Load faiss data
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f) 

#Loading the model embendig
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") 

def classify_document(text: str, top_k: int = 1):
    # text conversion to embending
    embedding = embedding_model.encode([text], normalize_embeddings=True).astype("float32")

    # compare the vector
    scores, indices = index.search(embedding, top_k)

    # save the data
    hits = []
    for idx, score in zip(indices[0], scores[0]):
        hits.append({
            "label": metadata[idx]["label"],
            "path": metadata[idx]["path"],
            "score": float(score)
        })

    # Send the data
    predicted_label = hits[0]["label"]
    confidence = hits[0]["score"]

    return predicted_label, confidence, hits
