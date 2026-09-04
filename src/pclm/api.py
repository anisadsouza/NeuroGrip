from __future__ import annotations

import time

import numpy as np
from fastapi import FastAPI, HTTPException

from pclm.features import FeatureConfig, extract_features, feature_vector
from pclm.model import PlaceholderGestureModel
from pclm.schemas import PredictionRequest, PredictionResponse, SignalQualityError
from pclm.signal_quality import check_signal_quality

app = FastAPI(title="Prosthetic Limb Gesture Control API", version="0.1.0")
model = PlaceholderGestureModel()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={422: {"model": SignalQualityError}},
)
def predict(request: PredictionRequest) -> PredictionResponse:
    started = time.perf_counter()
    channels = np.asarray(request.channels, dtype=np.float32)

    quality = check_signal_quality(channels)
    if not quality.ok:
        raise HTTPException(
            status_code=422,
            detail={"error": quality.reason, "failed_channel": quality.failed_channel},
        )

    config = FeatureConfig(sampling_rate_hz=request.sampling_rate_hz)
    features = extract_features(channels, config)
    prediction = model.predict(feature_vector(features), request.confidence_threshold)

    # Kept available for later latency logging without persisting raw EMG.
    _elapsed_ms = (time.perf_counter() - started) * 1000

    return PredictionResponse(
        subject_id=request.subject_id,
        label=prediction.label,
        confidence=prediction.confidence,
        abstained=prediction.abstained,
        feature_count=len(features),
    )


@app.get("/subject-info/{subject_id}")
def subject_info(subject_id: str) -> dict[str, object]:
    return {
        "subject_id": subject_id,
        "stored_raw_emg": False,
        "retained_fields": ["feature_count", "feature_statistics", "prediction_label", "confidence"],
    }
