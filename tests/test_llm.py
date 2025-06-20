import extractor as llm  # importa tu archivo real (ajusta el nombre si no es `llm.py`)

def test_build_prompt_includes_fields_and_text():
    document_type = "invoice"
    field_list = ["date", "total", "issuer"]
    text = "This is the OCR text from the document."

    prompt = llm.build_prompt(document_type, field_list, text)

    # Verificaciones básicas
    assert "invoice" in prompt
    assert "date" in prompt
    assert "total" in prompt
    assert "issuer" in prompt
    assert "This is the OCR text from the document." in prompt
    assert prompt.startswith("You are an intelligent extraction engine")

def test_build_payload_structure():
    prompt = "Sample prompt here"

    payload = llm.build_payload(prompt)

    assert isinstance(payload, dict)
    assert "model" in payload
    assert "messages" in payload
    assert "stream" in payload
    assert payload["messages"][0]["content"] == prompt
