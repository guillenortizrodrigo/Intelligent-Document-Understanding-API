from fastapi.testclient import TestClient
from pathlib import Path
import main

client = TestClient(main.app)

def test_full_flow_image(mocker, tmp_path):
    # ---------- 1) Mocks ----------
    mocker.patch("main.ocr_image", return_value="texto ocr")
    mocker.patch("main.classify_document", return_value=("invoice", 0.88, []))

    fake_entities = {
        "date":  {"value": "2024-05-01", "confidence": 0.97},
        "total": {"value": "$123.45",  "confidence": 0.93},
    }
    # ⬅️  devuelve tupla (dict, raw_json)
    mocker.patch(
        "main.extract_entities_with_ollama",
        return_value=(fake_entities, '{"date":"...", "total":"..."}')
    )

    # ---------- 2) Imagen dummy ----------
    img_path = tmp_path / "doc.jpg"
    img_path.write_bytes(b"\xFF\xD8\xFF")  # encabezado JPEG ficticio

    # ---------- 3) Llamada al endpoint ----------
    response = client.post(
        "/extract_entities/",
        files={"files": ("doc.jpg", img_path.read_bytes(), "image/jpeg")}
    )

    # ---------- 4) Verificaciones ----------
    assert response.status_code == 200
    body = response.json()

    assert body["results"][0]["document_type"] == "invoice"
    assert body["results"][0]["entities"] == fake_entities

    # ---------- 5) Verifica llamadas ----------
    main.ocr_image.assert_called_once()

    main.classify_document.assert_called_once()
    args, _ = main.classify_document.call_args
    assert args[0] == "texto ocr"

    main.extract_entities_with_ollama.assert_called_once()
    args, _ = main.extract_entities_with_ollama.call_args
    assert args[0] == "invoice"
    assert args[1] == "texto ocr"
