"""HTTP boundary: /api/v1/health, /ready, /model, /predictions."""

from __future__ import annotations

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.errors import ApiError
from app.schemas.prediction import DISCLAIMER, PredictionResponse
from app.services.prediction_service import predict_csv
from cardiomamba.data.preprocessing import ECGValidationError

router = APIRouter(prefix="/api/v1")


def _runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise ApiError(503, "MODEL_NOT_READY",
                       "The model is not loaded (checkpoint or thresholds unavailable).")
    return runtime


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "cardiomamba-api"}


@router.get("/ready")
def ready(request: Request) -> JSONResponse:
    ok = getattr(request.app.state, "runtime", None) is not None
    body = {"ready": ok}
    if not ok:
        body["reason"] = getattr(request.app.state, "load_error", "model not loaded")
    return JSONResponse(body, status_code=200 if ok else 503)


@router.get("/model")
def model_info(request: Request) -> dict:
    return {**_runtime(request).info, "disclaimer": DISCLAIMER}


@router.post("/predictions", response_model=PredictionResponse)
async def predictions(request: Request, file: UploadFile) -> PredictionResponse:
    runtime = _runtime(request)
    settings = request.app.state.settings
    name = (file.filename or "").lower()
    if not name.endswith(".csv"):
        raise ApiError(415, "UNSUPPORTED_FORMAT",
                       "Unsupported file type. Upload a .csv with 12 lead columns "
                       "(I, II, III, aVR, aVL, aVF, V1-V6) and 5000 rows (500 Hz) or "
                       "1000 rows (100 Hz).")
    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise ApiError(413, "PAYLOAD_TOO_LARGE",
                       f"File exceeds {settings.max_upload_bytes // (1024 * 1024)} MB.")
    try:
        return predict_csv(raw, runtime, request.state.request_id)
    except ECGValidationError as exc:
        raise ApiError(400, "INVALID_ECG_INPUT",
                       f"Unable to process this ECG: {exc}. Expected a 12-lead, 10-second "
                       "recording.") from exc
