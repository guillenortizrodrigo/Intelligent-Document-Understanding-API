from pathlib import Path
import numpy as np
from PIL import Image
import cv2 
import easyocr
import pdfplumber
from typing import Union
from tempfile import TemporaryDirectory

#load the model
reader = easyocr.Reader(['es', 'en'], gpu=False)

def ocr_image(path: Path) -> str:
    pre = preprocess_image(path)          # <─ nuevo paso 🔹
    results = reader.readtext(pre, detail=0, paragraph=True)
    return "\n".join(results)

def ocr_pdf(path: Path) -> str:
    with TemporaryDirectory() as tmpdir:
        pages_text = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                # rasterizar cada página a 300 dpi (~ buena para OCR)
                img = page.to_image(resolution=300).original
                img_path = Path(tmpdir) / f"page_{i}.png"
                img.save(img_path)
                pages_text.append(ocr_image(img_path))
        return "\n".join(pages_text).strip()


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
