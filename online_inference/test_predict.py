import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

CORRECT_DATA = [1, 120, 1, 2, 0, 3]
SHORT_DATA = [1, 0, 1]
INCORRECT_DATA =  [1, 120, 's', 2, 0, 3]


@pytest.mark.parametrize(
    'data_sample, response',
    [
        (CORRECT_DATA, 200),
        (SHORT_DATA, 400),
        (INCORRECT_DATA, 400),
    ]
)
def test_predict(data_sample, response):
    response = client.get("/predict/", json={'data': [data_sample]})
    return response.status_code