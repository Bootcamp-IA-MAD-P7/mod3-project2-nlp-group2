import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "model"))

import numpy as np
import pandas as pd
import pytest
import joblib
import train  # noqa: F401
from train import MultiLabelPipeline
sys.modules['__main__'].MultiLabelPipeline = MultiLabelPipeline

MODEL_PATH = "models/tfidf_lr.joblib"
TEST_CSV = "data/processed/test.csv"
NON_LABEL_COLS = {"CommentId", "clean_text"}
LABELS = [c for c in pd.read_csv(TEST_CSV, nrows=0).columns if c not in NON_LABEL_COLS]


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(MODEL_PATH)


def test_labels_count(pipeline):
    """El modelo predice exactamente 12 etiquetas."""
    pred = pipeline.predict(["you are stupid"])
    assert pred.shape[1] == 12


def test_labels_order(pipeline):
    """Las etiquetas del modelo están en el orden correcto."""
    assert pipeline.labels == LABELS


def test_output_binary(pipeline):
    """Las predicciones son solo 0 o 1."""
    pred = pipeline.predict(["I hate you", "have a nice day"])
    assert set(np.unique(pred)).issubset({0, 1})


def test_output_shape(pipeline):
    """Shape de salida es (n_comentarios, 12)."""
    texts = ["hello", "you are trash", "go away"]
    pred = pipeline.predict(texts)
    assert pred.shape == (3, 12)


def test_toxic_comment_detected(pipeline):
    """Un comentario claramente tóxico activa IsToxic."""
    pred = pipeline.predict(["I will kill you, you disgusting piece of trash"])
    is_toxic_idx = LABELS.index("IsToxic")
    assert pred[0][is_toxic_idx] == 1
