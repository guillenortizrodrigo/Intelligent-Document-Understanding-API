import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# === Rutas de los archivos guardados ===
INDEX_PATH = "vector_index.faiss"
META_PATH = "metadata.pkl"

# === Carga de FAISS y metadata ===
print("📦 Cargando índice vectorial y metadata...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)  # Lista de dicts con 'label', 'text', etc.

# === Cargar modelo de embeddings ===
print("🧠 Cargando modelo de embeddings...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # ¡usa el mismo modelo!

def classify_document(text: str, top_k: int = 3):
    """
    Recibe un texto y devuelve el tipo de documento más probable usando similitud semántica.

    Args:
        text: Texto extraído con OCR
        top_k: Número de documentos similares a devolver (por default 3)

    Returns:
        doc_type: etiqueta más probable (ej. 'invoice')
        confidence: score (similaridad coseno) del top-1
        hits: lista con los top_k resultados
    """
    # 1. Convertir texto a embedding
    embedding = embedding_model.encode([text], normalize_embeddings=True).astype("float32")

    # 2. Buscar en FAISS
    scores, indices = index.search(embedding, top_k)  # (1, top_k)

    # 3. Empaquetar resultados
    hits = []
    for idx, score in zip(indices[0], scores[0]):
        hits.append({
            "label": metadata[idx]["label"],
            "path": metadata[idx]["path"],
            "score": float(score)
        })

    # 4. Tomar el top-1 como predicción
    predicted_label = hits[0]["label"]
    confidence = hits[0]["score"]

    return predicted_label, confidence, hits
