"""Response schemas (System-Design/08-api-contracts/01-prediction-api.md)."""

from __future__ import annotations

from pydantic import BaseModel

DISCLAIMER = ("Research/educational prototype. Predictions are model outputs and are not "
              "medical diagnoses.")


class ModelSummary(BaseModel):
    name: str
    version: str
    task: str
    preprocessing_version: str
    checkpoint_epoch: int


class InputSummary(BaseModel):
    format: str
    leads: int
    samples: int
    sampling_rate_hz: int
    model_sampling_rate_hz: int


class LabelPrediction(BaseModel):
    label: str
    probability: float
    threshold: float
    positive: bool


class Waveform(BaseModel):
    sampling_rate_hz: int
    units: str
    leads: list[str]
    data: list[list[float]]           # [12][1000], resampled, un-normalized (for display)


class PredictionResponse(BaseModel):
    request_id: str
    model: ModelSummary
    input: InputSummary
    predictions: list[LabelPrediction]
    waveform: Waveform
    inference_ms: float
    device: str
    disclaimer: str = DISCLAIMER


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
