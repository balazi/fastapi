import numpy as np
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    load_artifacts,
    save_artifacts,
    train_model,
)


def _toy():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 25, 35, 45, 55],
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
        "salary": ["<=50K", ">50K"] * 4,
    })
    X, y, enc, lb = process_data(
        df, categorical_features=["sex"], label="salary", training=True
    )
    return df, X, y, enc, lb


def test_train_and_inference():
    _, X, y, _, _ = _toy()
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape
    assert set(preds) <= {0, 1}


def test_slice_metrics_and_artifacts(tmp_path):
    df, X, y, enc, lb = _toy()
    model = train_model(X, y)
    save_artifacts(model, enc, lb, tmp_path)
    model2, enc2, lb2 = load_artifacts(tmp_path)
    out = compute_slice_metrics(df, "sex", model2, enc2, lb2, ["sex"])
    assert list(out["value"]) == ["F", "M"]
    assert out["count"].sum() == len(df)


def test_compute_model_metrics_perfect():
    y = np.array([0, 1, 1, 0])
    assert compute_model_metrics(y, y) == (1.0, 1.0, 1.0)


def test_process_data_shapes():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "sex": ["M", "F", "M"],
        "salary": ["<=50K", ">50K", ">50K"],
    })
    X, y, _, _ = process_data(
        df, categorical_features=["sex"], label="salary", training=True
    )
    assert X.shape[0] == 3
    assert y.shape == (3,)
