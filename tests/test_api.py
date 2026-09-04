from fastapi.testclient import TestClient

from pclm.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_prediction():
    response = client.post(
        "/predict",
        json={
            "subject_id": "demo",
            "sampling_rate_hz": 2000,
            "channels": [
                [0.1, 0.2, -0.1, 0.3, -0.2],
                [0.2, 0.1, -0.2, 0.1, -0.1],
            ],
            "confidence_threshold": 0.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["subject_id"] == "demo"
    assert body["feature_count"] > 0


def test_predict_endpoint_rejects_bad_signal():
    response = client.post(
        "/predict",
        json={
            "subject_id": "demo",
            "sampling_rate_hz": 2000,
            "channels": [[0.0, 0.0, 0.0]],
        },
    )

    assert response.status_code == 422
