"""Validation -> production preprocessing -> inference -> thresholding."""

from __future__ import annotations

import time

import numpy as np
import torch

from app.ml.decoding import decode_csv
from app.ml.model_loader import ModelRuntime
from app.schemas.prediction import (
    InputSummary,
    LabelPrediction,
    ModelSummary,
    PredictionResponse,
    Waveform,
)
from cardiomamba.data.labels import CLASS_NAMES
from cardiomamba.data.preprocessing import LEAD_NAMES, TARGET_FS, resample_to_target


def predict_csv(raw: bytes, runtime: ModelRuntime, request_id: str) -> PredictionResponse:
    signal, fs = decode_csv(raw)                        # validated [T, 12], canonical order
    resampled = resample_to_target(signal, fs)          # prep-v1: [1000, 12] mV at 100 Hz
    x = runtime.stats.apply(resampled)                  # frozen training-fold normalization
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = runtime.model(torch.from_numpy(x).unsqueeze(0).to(runtime.device))
        probs = torch.sigmoid(logits.float())[0].cpu().numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    info = runtime.info
    return PredictionResponse(
        request_id=request_id,
        model=ModelSummary(name=info["name"], version=info["version"], task=info["task"],
                           preprocessing_version=info["preprocessing_version"],
                           checkpoint_epoch=info["checkpoint_epoch"]),
        input=InputSummary(format="csv", leads=signal.shape[1], samples=signal.shape[0],
                           sampling_rate_hz=fs, model_sampling_rate_hz=TARGET_FS),
        predictions=[
            LabelPrediction(label=c, probability=float(p), threshold=runtime.thresholds[c],
                            positive=bool(p >= runtime.thresholds[c]))
            for c, p in zip(CLASS_NAMES, probs, strict=True)
        ],
        waveform=Waveform(sampling_rate_hz=TARGET_FS, units="mV", leads=list(LEAD_NAMES),
                          data=np.round(resampled.T.astype(np.float64), 4).tolist()),
        inference_ms=round(elapsed_ms, 1),
        device=str(runtime.device),
    )
