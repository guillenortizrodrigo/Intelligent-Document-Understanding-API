from pathlib import Path
import numpy as np
from PIL import Image
import easyocr
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# === Configuración ===
BASE_DIR = Path("docs-sm")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
OUTPUT_INDEX = "vector_index.faiss"
OUTPUT_METADATA = "metadata.pkl"

# === Modelos ===
reader = easyocr.Reader(['en', 'es'], gpu=False)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Variables para FAISS ===
embeddings_list = []
metadata_list = []

# === Procesar documentos ===
for label_dir in BASE_DIR.iterdir():
    if label_dir.is_dir():
        label = label_dir.name  # Ej: 'invoice', 'memo', etc.
        for file in label_dir.glob("*"):
            if file.suffix.lower() in ALLOWED_EXTENSIONS:
                try:
                    print(f"OCR → {file}")
                    img = np.array(Image.open(file))
                    text = "\n".join(reader.readtext(img, detail=0, paragraph=True))
                    if not text.strip():
                        continue
                    emb = embedding_model.encode([text], normalize_embeddings=True)
                    embeddings_list.append(emb[0])
                    metadata_list.append({
                        "path": str(file),
                        "label": label,
                        "text": text
                    })
                except Exception as e:
                    print(f"Error procesando {file}: {e}")

# === Guardar en FAISS ===
if embeddings_list:
    embeddings_np = np.array(embeddings_list, dtype="float32")
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine sim (si normalizas)
    index.add(embeddings_np)

    faiss.write_index(index, OUTPUT_INDEX)
    with open(OUTPUT_METADATA, "wb") as f:
        pickle.dump(metadata_list, f)

    print("✅ FAISS y metadata guardados.")
else:
    print("⚠️ No se encontraron documentos válidos.")
