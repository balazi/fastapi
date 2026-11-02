"""Script to train the model, save artifacts and compute slice metrics."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    save_artifacts,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "census.csv"
MODEL_DIR = ROOT / "model"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def main():
    data = pd.read_csv(DATA_PATH)
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary",
        training=False, encoder=encoder, lb=lb,
    )

    model = train_model(X_train, y_train)
    save_artifacts(model, encoder, lb, MODEL_DIR)

    precision, recall, fbeta = compute_model_metrics(
        y_test, inference(model, X_test))
    print(f"precision={precision:.4f} recall={recall:.4f} fbeta={fbeta:.4f}")

    slices = pd.concat([
        compute_slice_metrics(
            test, feature, model, encoder, lb, CAT_FEATURES)
        for feature in CAT_FEATURES
    ])
    report = slices.to_string(index=False)
    print(report)
    out = MODEL_DIR / "slice_output.txt"
    out.write_text(report + "\n")
    print(f"Slice metrics written to {out}")


if __name__ == "__main__":
    main()
