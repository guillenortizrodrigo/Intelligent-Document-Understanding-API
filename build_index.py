from pathlib import Path
import numpy as np
from PIL import Image
import easyocr
from sentence_transformers import SentenceTransformer
import faiss
import os
import cv2
import pickle
from typing import Union
import torch

BASE_DIR = Path("docs-sm")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
OUTPUT_INDEX = "vector_index.faiss"
OUTPUT_METADATA = "metadata.pkl"

#models
reader = easyocr.Reader(['en', 'es'], gpu=True)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings_list = []
metadata_list = []

processed_one = False    


def preprocess_image(path: Union[str, Path]) -> np.ndarray:
    # --- 1. Leer y a gris ---
    img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # --- 2. Suavizado / eliminación de ruido ---
    denoised = cv2.fastNlMeansDenoising(img_gray, h=15, templateWindowSize=7,
                                        searchWindowSize=21)

    # --- 3. Binarización adaptativa ---
    bin_img = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31,   # cuanto más grande, más contexto usa
        C=10            # constante que ajusta el umbral
    )

    # --- 4. Deskew ---
    coords = np.column_stack(np.where(bin_img == 0))
    angle = cv2.minAreaRect(coords)[-1]
    # cv2 devuelve ángulos en (-90, 0]; convertimos a [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    h, w = bin_img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(bin_img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return deskewed


#process all files
for label_dir in BASE_DIR.iterdir():
    if label_dir.is_dir():
        label = label_dir.name
        for file in label_dir.glob("*"):
            if file.suffix.lower() in ALLOWED_EXTENSIONS:
                try:
                    print(f"OCR → {file}")
                    img = preprocess_image(file) 
                    #img = np.array(Image.open(file))
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

#save vector on FAIIS
if embeddings_list:
    embeddings_np = np.array(embeddings_list, dtype="float32")
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_np)

    faiss.write_index(index, OUTPUT_INDEX)
    with open(OUTPUT_METADATA, "wb") as f:
        pickle.dump(metadata_list, f)

    print("FAISS y metadata guardados.")
else:
    print("No se encontraron documentos válidos.")



