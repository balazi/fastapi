from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

LOW_INCOME = {
    "age": 39, "workclass": "State-gov", "fnlgt": 77516,
    "education": "Bachelors", "education-num": 13,
    "marital-status": "Never-married", "occupation": "Adm-clerical",
    "relationship": "Not-in-family", "race": "White", "sex": "Male",
    "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40,
    "native-country": "United-States",
}

HIGH_INCOME = {
    "age": 52, "workclass": "Self-emp-not-inc", "fnlgt": 209642,
    "education": "Doctorate", "education-num": 16,
    "marital-status": "Married-civ-spouse", "occupation": "Prof-specialty",
    "relationship": "Husband", "race": "White", "sex": "Male",
    "capital-gain": 15024, "capital-loss": 0, "hours-per-week": 60,
    "native-country": "United-States",
}


def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "message": "Welcome to the census income prediction API"}


def test_post_predict_low_income():
    r = client.post("/predict", json=LOW_INCOME)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_predict_high_income():
    r = client.post("/predict", json=HIGH_INCOME)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
