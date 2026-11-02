"""Make one POST request to the live API and print the result."""
import os

import requests

URL = os.environ.get(
    "API_URL", "https://fastapi-16n5.onrender.com") + "/predict"

SAMPLE = {
    "age": 52, "workclass": "Self-emp-not-inc", "fnlgt": 209642,
    "education": "Doctorate", "education-num": 16,
    "marital-status": "Married-civ-spouse", "occupation": "Prof-specialty",
    "relationship": "Husband", "race": "White", "sex": "Male",
    "capital-gain": 15024, "capital-loss": 0, "hours-per-week": 60,
    "native-country": "United-States",
}

if __name__ == "__main__":
    response = requests.post(URL, json=SAMPLE, timeout=60)
    print("Status code:", response.status_code)
    print("Result:", response.json())
