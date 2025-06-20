from fastapi.testclient import TestClient
from main import app, allowed_file
import io
from unittest.mock import patch

client = TestClient(app)

def test_allowed_file():
    assert allowed_file("file.pdf")
    assert not allowed_file("file.exe")

@patch("main.ocr_image", return_value="   ")  # el OCR no encontró nada
def test_invalid_file_content(mock_ocr):
    file = io.BytesIO(b"contenido falso")
    response = client.post(
        "/extract_entities/",
        files={"files": ("test.png", file, "image/png")}
    )
    assert response.status_code == 415
    assert "No legible text" in response.json()["detail"]


@patch("main.ocr_image", return_value="texto extraído del documento")
@patch("main.classify_document", return_value=("invoice", 0.95, None))
@patch("main.extract_entities_with_ollama", return_value={"nombre": "Pedro"})
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
@patch("main.extract_entities_with_ollama", return_value={"asunto": "reunión"})
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

