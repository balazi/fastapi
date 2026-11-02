from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save_artifacts(model, encoder, lb, directory):
    """Save the model, encoder and label binarizer into directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib", compress=3)
    joblib.dump(encoder, directory / "encoder.joblib")
    joblib.dump(lb, directory / "lb.joblib")


def load_artifacts(directory):
    """Load (model, encoder, lb) saved by save_artifacts."""
    directory = Path(directory)
    return (
        joblib.load(directory / "model.joblib"),
        joblib.load(directory / "encoder.joblib"),
        joblib.load(directory / "lb.joblib"),
    )


def compute_slice_metrics(df, feature, model, encoder, lb,
                          categorical_features, label="salary"):
    """Compute precision, recall and F1 for each value of a categorical
    feature.

    Returns
    -------
    pd.DataFrame with columns feature, value, count, precision, recall,
    fbeta.
    """
    rows = []
    for value in sorted(df[feature].unique()):
        subset = df[df[feature] == value]
        X, y, _, _ = process_data(
            subset, categorical_features=categorical_features, label=label,
            training=False, encoder=encoder, lb=lb,
        )
        precision, recall, fbeta = compute_model_metrics(
            y, inference(model, X))
        rows.append({
            "feature": feature, "value": value, "count": len(subset),
            "precision": precision, "recall": recall, "fbeta": fbeta,
        })
    return pd.DataFrame(rows)
