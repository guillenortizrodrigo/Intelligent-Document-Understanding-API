import numpy as np
import pytest
import classifier as clf
from pytest import approx

def test_classify_document_returns_label_confidence_hits(mocker):
    test_text = "Sample document text"

    # Mock del modelo de embeddings
    dummy_embedding = np.array([[0.1, 0.2, 0.3]])  # ya es array
    mocker.patch.object(clf.embedding_model, "encode", return_value=dummy_embedding)

    # Mock del índice FAISS
    dummy_scores = np.array([[0.95]], dtype="float32")
    dummy_indices = np.array([[0]])
    mock_index = mocker.patch.object(clf.index, "search", return_value=(dummy_scores, dummy_indices))

    # Mock del metadata
    clf.metadata[0] = {
        "label": "invoice",
        "path": "docs/invoice-123.jpg"
    }

    label, confidence, hits = clf.classify_document(test_text)

    # Verificaciones
    clf.embedding_model.encode.assert_called_once_with([test_text], normalize_embeddings=True)
    mock_index.assert_called_once()
    assert label == "invoice"
    assert confidence == approx(0.95, rel=1e-6)
    assert hits[0]["label"] == "invoice"
    assert hits[0]["path"] == "docs/invoice-123.jpg"
    assert hits[0]["score"] == approx(0.95, rel=1e-6)
