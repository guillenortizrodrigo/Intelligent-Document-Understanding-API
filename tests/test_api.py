from fastapi.testclient import TestClient
from main import app, allowed_file
import io
from unittest.mock import patch

client = TestClient(app)

# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #
def assert_error_response(response, expected_code, error_code, expected_message_substring):
    assert response.status_code == expected_code
    detail = response.json()["detail"]
    assert detail["error"] == error_code
    assert expected_message_substring in detail["message"]

# -------------------------------------------------------------------- #
# Unit tests
# -------------------------------------------------------------------- #
def test_allowed_file():
    assert allowed_file("file.pdf")
    assert not allowed_file("file.exe")


@patch("main.ocr_image", return_value="   ")
def test_invalid_file_content(mock_ocr):
    file = io.BytesIO(b"contenido falso")
    response = client.post(
        "/extract_entities/",
        files={"files": ("test.png", file, "image/png")}
    )
    assert_error_response(response, 415, "NoTextFound", "No legible text")


@patch("main.ocr_image", return_value="texto extraído del documento")
@patch("main.classify_document", return_value=("invoice", 0.95, None))
@patch("main.extract_entities_with_ollama",
       return_value=({"nombre": "Pedro"}, '{"nombre": "Pedro"}'))   # ⬅️ tupla
def test_successful_extraction(mock_ollama, mock_classify, mock_ocr):
    file = io.BytesIO(b"imagen falsa pero pasa")
    response = client.post(
        "/extract_entities/",
        files={"files": ("factura.jpg", file, "image/jpeg")}
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["document_type"] == "invoice"
    assert result["confidence"] == 0.95
    assert result["entities"]["nombre"] == "Pedro"


@patch("main.ocr_image", return_value="texto simulado")
@patch("main.classify_document", return_value=("memo", 0.87, None))
@patch("main.extract_entities_with_ollama",
       return_value=({"asunto": "reunión"}, '{"asunto": "reunión"}'))  # ⬅️ tupla
def test_multiple_files(mock_ollama, mock_classify, mock_ocr):
    file1 = io.BytesIO(b"contenido archivo 1")
    file2 = io.BytesIO(b"contenido archivo 2")

    files = [
        ("files", ("memo1.jpg", file1, "image/jpeg")),
        ("files", ("memo2.jpg", file2, "image/jpeg")),
    ]

    response = client.post("/extract_entities/", files=files)

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["document_type"] == "memo"
    assert results[1]["entities"]["asunto"] == "reunión"
