# extractor.py
"""
Extrae entidades con un modelo local (llama3) sirviendo vía Ollama.
Devuelve: entidades (dict) | error (str) | tiempo de proceso (float s)
"""

import requests
import time
import json, pathlib

# Constants

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"
SCHEMA_PATH = pathlib.Path(__file__).with_name("document_schema.json")

# Cárgalo una sola vez al iniciar el módulo
with SCHEMA_PATH.open(encoding="utf-8") as f:
    DOCUMENT_SCHEMA: dict[str, list[str]] = json.load(f)


def build_prompt(document_type: str, field_list: list[str], text: str) -> str:
    return (
        f"Given the following text extracted from a document of type "
        f"'{document_type}', extract these fields: {', '.join(field_list)}. if you don't find the field just write not found "
        "Return your response as a valid JSON object with no additional text.\n\n"
        "Document Text:\n"
        f"{text}"
    )

def build_payload(prompt):
    messages = [{"role": "user", "content": prompt}]
    return {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "format": "json" 
    }

def extract_entities_with_ollama(document_type: str, document_text: str):
    t0 = time.perf_counter()

    # obtener la lista de los campos
    field_list = DOCUMENT_SCHEMA.get(document_type)
    if not field_list:
        raise ValueError(f"[extract_entities] Document type '{document_type}' no está definido en document_schema.json")

    # construir el payload
    prompt = build_prompt(document_type, field_list, document_text)
    payload = build_payload(prompt)

    response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
    print(response.json()['message']['content'])
    raw = response.json()["message"]["content"]
    try:
        entities = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ollama devolvió JSON inválido: {e}\nRAW:\n{raw}")

    return entities, time.perf_counter() - t0




