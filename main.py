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
import time
import json
from logging_setup import logger
import uuid

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
UPLOAD_DIR = "uploads"

app = FastAPI(title="Entity Extraction API")

def allowed_file(filename: str) -> bool:
    return filename.split(".")[-1].lower() in ALLOWED_EXTENSIONS

def logger_log(message, type, trace_id, file, phase, error=""):
    log_data = {
        "trace_id": trace_id,
        "file": str(file),
        "phase": phase,
        "error": str(error)
    }
    if type == "error":
        logger.error(message, extra=log_data, exc_info=True)
    elif type == "warning":
        logger.warning(message, extra=log_data)
    elif type == "info":
        logger.info(message, extra=log_data)


async def process_file(file_path: str) -> dict:
    trace_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    ext = Path(file_path).suffix.lower()

    # ---------- OCR ----------
    try:
        if ext == ".pdf":
            logger_log("Processing PDF file", "info", trace_id, file_path, "ocr")
            text = ocr_pdf(Path(file_path))
        else:
            logger_log("Processing image file", "info", trace_id, file_path, "ocr")
            text = ocr_image(Path(file_path))
    except Exception as e:
        logger_log("OCR failed", "error", trace_id, file_path, "ocr", e)
        raise HTTPException(status_code=500, detail={
            "error": "OCRFailure",
            "message": "OCR processing failed",
            "trace_id": trace_id
        })

    if not text.strip():
        logger_log("No text found in document", "warning", trace_id, file_path, "ocr")
        raise HTTPException(status_code=415, detail={
            "error": "NoTextFound",
            "message": "No legible text found in document",
            "trace_id": trace_id
        })

    # ---------- Classification ----------
    try:
        logger_log("Classifying document", "info", trace_id, file_path, "classification")
        doc_type, confidence, hits = classify_document(text)
    except Exception as e:
        logger_log("Classification failed", "error", trace_id, file_path, "classification", e)
        raise HTTPException(status_code=500, detail={
            "error": "ClassificationError",
            "message": "Document classification failed",
            "trace_id": trace_id
        })

    # ---------- LLM Extraction ----------
    try:
        logger_log("Extracting entities using LLM", "info", trace_id, file_path, "llm")
        entities = extract_entities_with_ollama(doc_type, text)
    except json.JSONDecodeError as e:
        logger_log("LLM returned malformed JSON", "warning", trace_id, file_path, "llm", e)
        raise HTTPException(status_code=502, detail={
            "error": "LLMResponseInvalid",
            "message": "The LLM returned malformed JSON",
            "hint": "Retry with lower temperature or validate model behavior",
            "trace_id": trace_id
        })
    except Exception as e:
        logger_log("LLM extraction failed", "error", trace_id, file_path, "llm", e)
        raise HTTPException(status_code=500, detail={
            "error": "LLMError",
            "message": "Entity extraction failed",
            "trace_id": trace_id
        })

    processing_time = time.perf_counter() - t0

    logger.info("File Process Completed", extra={
        "trace_id": trace_id,
        "file": str(file_path),
        "time": str(processing_time),
    })

    return {
        "filename": Path(file_path).name,
        "document_type": doc_type,
        "confidence": round(confidence, 2),
        "entities": entities,
        "processing_time": processing_time,
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

