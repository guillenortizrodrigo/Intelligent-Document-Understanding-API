# tests/test_ocr.py
from pathlib import Path
import numpy as np
import types

import pytest
import ocr as ocr_module
from unittest.mock import MagicMock

# ---------- helpers ----------
def fake_image(w=100, h=50, value=255):
    """Devuelve un ndarray 'imagen' sencilla."""
    return np.full((h, w), value, dtype=np.uint8)

# ---------- tests preprocess_image ----------
def test_preprocess_image_returns_ndarray(tmp_path):
    # Crea una imagen temporal completamente blanca
    img_path = tmp_path / "white.png"
    import cv2
    cv2.imwrite(str(img_path), fake_image())

    processed = ocr_module.preprocess_image(img_path)

    assert isinstance(processed, np.ndarray)
    # misma forma
    assert processed.shape == (50, 100)

def test_preprocess_image_no_crash_on_blank(tmp_path):
    img_path = tmp_path / "blank.png"
    import cv2
    cv2.imwrite(str(img_path), fake_image())

    # No debe lanzar excepción aunque la imagen sea “vacía”
    ocr_module.preprocess_image(img_path)


# ---------- tests ocr_image ----------
def test_ocr_image_calls_reader_and_preprocess(mocker, tmp_path):
    img_path = tmp_path / "foo.png"
    import cv2
    cv2.imwrite(str(img_path), fake_image())

    # 1) Mock preprocess_image ⇒ devuelve imagen “lista”
    dummy_processed = fake_image()
    mocker.patch("ocr.preprocess_image", return_value=dummy_processed)

    # 2) Mock reader.readtext ⇒ devuelve lista simulada
    fake_reader = mocker.patch.object(
        ocr_module.reader, "readtext", return_value=["Hola", "mundo"]
    )

    text = ocr_module.ocr_image(img_path)

    fake_reader.assert_called_once_with(dummy_processed, detail=0, paragraph=True)
    assert text == "Hola\nmundo"


# ---------- tests ocr_pdf ----------
class DummyPage:
    def __init__(self, idx):
        self.idx = idx

    def to_image(self, resolution):
        # Devuelve objeto con atributo .original como PIL.Image
        from PIL import Image
        arr = np.full((10, 10), 255, dtype=np.uint8)
        return types.SimpleNamespace(original=Image.fromarray(arr))

def test_ocr_pdf_iterates_pages_and_concatenates(mocker, tmp_path):
    # 1) Simular las páginas del PDF
    dummy_pages = [DummyPage(1), DummyPage(2)]

    # 2) Crear un objeto mock que simula el contexto 'with'
    dummy_pdf_context = MagicMock()
    dummy_pdf_context.__enter__.return_value.pages = dummy_pages
    mocker.patch("ocr.pdfplumber.open", return_value=dummy_pdf_context)

    # 3) Mock ocr_image => texto por página
    mocker.patch("ocr.ocr_image", side_effect=["pag1", "pag2"])

    # 4) Crear archivo dummy
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")  # contenido ficticio de PDF

    result = ocr_module.ocr_pdf(pdf_path)

    # 5) Verificaciones
    assert result == "pag1\npag2"
