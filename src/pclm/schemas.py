from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    subject_id: str = Field(..., min_length=1)
    sampling_rate_hz: int = Field(2000, ge=100, le=10000)
    channels: list[list[float]] = Field(..., min_length=1)
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)

    @field_validator("channels")
    @classmethod
    def channels_must_have_same_length(cls, channels: list[list[float]]) -> list[list[float]]:
        lengths = {len(channel) for channel in channels}
        if len(lengths) != 1:
            raise ValueError("all channels must have the same sample count")
        if min(lengths) < 3:
            raise ValueError("each channel must contain at least three samples")
        return channels


class PredictionResponse(BaseModel):
    subject_id: str
    label: str
    confidence: float
    abstained: bool
    feature_count: int


class SignalQualityError(BaseModel):
    error: str
    failed_channel: int | None = None
