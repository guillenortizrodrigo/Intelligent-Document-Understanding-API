from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import aiofiles
import os
import uuid
from pathlib import Path
from ocr import ocr_image, ocr_pdf
from classifier import classify_document
from extractor import extract_entities_with_ollama

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
UPLOAD_DIR = "uploads"

app = FastAPI(title="Entity Extraction API")

def allowed_file(filename: str) -> bool:
    return filename.split(".")[-1].lower() in ALLOWED_EXTENSIONS

async def process_file(file_path: str) -> dict:
    ext = Path(file_path).suffix.lower()

    # ---------- OCR ----------
    if ext == ".pdf":
        text = ocr_pdf(Path(file_path))
    else:
        text = ocr_image(Path(file_path))

    if not text.strip():
        raise HTTPException(
            status_code=415,
            detail="No legible text found in document"
        )
        
    # ---------- Clasificación semántica ----------
    doc_type, confidence, hits = classify_document(text)

    # ---------- Entities ----------
    entities = extract_entities_with_ollama(doc_type,text)

    return {
        "filename": Path(file_path).name,
        "document_type": doc_type,
        "confidence": round(confidence, 2),
        "entities":entities
    }

@app.post("/extract_entities/")
async def extract_entities(files: List[UploadFile] = File(...)):
    responses = []

    for file in files:
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail=f"Not allowed format: {file.filename}")

        # Create the Upload directory if it is necessary
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # temporarily save the file
        temp_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)

        async with aiofiles.open(temp_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        # Procesing of the file
        try:
            result = await process_file(temp_path)
            responses.append(result)
        finally:
            # Delete the file after
            os.remove(temp_path)

    return JSONResponse(content={"results": responses})

