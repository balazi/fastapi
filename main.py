"""FastAPI application serving the census income model."""
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_artifacts

MODEL_DIR = Path(__file__).resolve().parent / "model"

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

model, encoder, lb = load_artifacts(MODEL_DIR)

app = FastAPI(title="Census Income Prediction API")


class CensusRecord(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.get("/")
async def root() -> dict:
    return {"message": "Welcome to the census income prediction API"}


@app.post("/predict")
async def predict(record: CensusRecord) -> dict:
    df = pd.DataFrame([record.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        df, categorical_features=CAT_FEATURES, training=False,
        encoder=encoder, lb=lb,
    )
    pred = inference(model, X)
    return {"prediction": str(lb.inverse_transform(pred)[0])}
