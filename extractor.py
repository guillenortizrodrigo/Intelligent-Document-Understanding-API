import requests
import json, pathlib
import os

# Constants

OLLAMA_API = os.getenv("OLLAMA_API","http://localhost:11434/api/chat")
HEADERS = {"Content-Type": "application/json"}
MODEL_OLLAMA = os.getenv("OLLAMA_MODEL","llama3.2")
SCHEMA_PATH = pathlib.Path(__file__).with_name("document_schema.json")


with SCHEMA_PATH.open(encoding="utf-8") as f:
    DOCUMENT_SCHEMA: dict[str, list[str]] = json.load(f)


def build_prompt(document_type: str, field_list: list[str], text: str) -> str:
    fields = ", ".join(field_list)
    return (
        f"You are an intelligent extraction engine designed to process {document_type} documents.\n"
        f"For each of the following fields — {fields} — extract the value and assign a confidence score "
        f"between 0 and 1, based on how certain you are about the value.\n\n"
        "Respond with a **valid JSON object** where each field name maps to an object with two keys:\n"
        "- `value`: the extracted value (string or list of strings)\n"
        "- `confidence`: a float between 0 and 1 indicating your confidence in the extracted value\n\n"
        "If a value is not found, set:\n"
        "- `value`: \"not found\"\n"
        "- `confidence`: 0.0\n\n"
        "Example format:\n"
        "{\n"
        "  \"field_name\": {\n"
        "    \"value\": \"example value\",\n"
        "    \"confidence\": 0.92\n"
        "  },\n"
        "  ...\n"
        "}\n\n"
        "Now extract the information from the following text:\n\n"
        f"{text}"
    )


def build_payload(prompt):
    messages = [{"role": "user", "content": prompt}]
    return {
        "model": MODEL_OLLAMA,
        "messages": messages,
        "stream": False,
        "format": "json" 
    }

def extract_entities_with_ollama(document_type: str, document_text: str):
    field_list = DOCUMENT_SCHEMA.get(document_type)
    if not field_list:
        raise ValueError(f"'{document_type}' no está definido en document_schema.json")

    prompt   = build_prompt(document_type, field_list, document_text)
    payload  = build_payload(prompt)

    resp = requests.post(OLLAMA_API, json=payload, headers=HEADERS).json()
    print(resp["message"])
    raw  = resp["message"]["content"]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido:\n{raw}") from e

    # Asegúrate de que todos los campos existan
    result = {
        f: data.get(f, {"value": "not found", "confidence": 0.0})
        for f in field_list
    }
    
    return result , raw





