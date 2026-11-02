import numpy as np
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics


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
