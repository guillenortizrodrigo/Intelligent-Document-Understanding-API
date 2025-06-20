from pathlib import Path
import numpy as np
from PIL import Image
import easyocr
import pdfplumber

#load the model
reader = easyocr.Reader(['es', 'en'], gpu=False)

def ocr_image(path: Path) -> str:
    img = np.array(Image.open(path))
    results = reader.readtext(img, detail=0, paragraph=True)
    return "\n".join(results)

def ocr_pdf(path: Path) -> str:
    with pdfplumber.open(str(path)) as pdf:
        text = "\n".join(
            page.extract_text() or "" for page in pdf.pages
        )
    return text.strip()
