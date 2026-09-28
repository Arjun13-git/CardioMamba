"""API tests. Inference tests need the frozen checkpoint and skip when it is absent."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import FROZEN_RUN, REPO_ROOT, Settings
from app.main import create_app

DEMO_DIR = REPO_ROOT / "frontend" / "public" / "demo"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
EXPECTED_THRESHOLDS = {"NORM": 0.38, "MI": 0.26, "STTC": 0.41, "CD": 0.35, "HYP": 0.27}
needs_model = pytest.mark.skipif(not (FROZEN_RUN / "best.pt").is_file(),
                                 reason="frozen checkpoint not available")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(Settings())) as c:
        yield c


def csv_bytes(rows: int = 5000, header=LEADS, fill=None) -> bytes:
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.2, size=(rows, len(header))) if fill is None else fill
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(data.tolist())
    return buf.getvalue().encode()


def upload(client, content: bytes, name: str = "ecg.csv"):
    return client.post("/api/v1/predictions", files={"file": (name, content, "text/csv")})


def test_health(client):
    assert client.get("/api/v1/health").json() == {"status": "ok", "service": "cardiomamba-api"}


@needs_model
def test_ready_and_model_info(client):
    assert client.get("/api/v1/ready").json() == {"ready": True}
    info = client.get("/api/v1/model").json()
    assert info["class_order"] == ["NORM", "MI", "STTC", "CD", "HYP"]
    assert info["thresholds"] == EXPECTED_THRESHOLDS
    assert info["parameters"] == 939_397 and info["checkpoint_epoch"] == 5
    assert info["test_metrics"]["n_records"] == 2158
    assert "not medical diagnoses" in info["disclaimer"]


def test_model_not_ready_is_structured_503(tmp_path):
    settings = Settings(checkpoint_path=tmp_path / "missing.pt")
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/v1/ready").status_code == 503
        r = upload(c, csv_bytes())
        assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_READY"


@needs_model
@pytest.mark.parametrize(("content", "name", "status", "code"), [
    (b"hello", "ecg.txt", 415, "UNSUPPORTED_FORMAT"),
    (csv_bytes(rows=4000), "ecg.csv", 400, "INVALID_ECG_INPUT"),
    (csv_bytes(header=LEADS[:11] + ["V7"]), "ecg.csv", 400, "INVALID_ECG_INPUT"),
    (b"I,II\n1,2\n", "ecg.csv", 400, "INVALID_ECG_INPUT"),
    (csv_bytes().replace(b"\n0", b"\nabc", 1), "ecg.csv", 400, "INVALID_ECG_INPUT"),
    (csv_bytes(fill=np.full((5000, 12), np.nan)), "ecg.csv", 400, "INVALID_ECG_INPUT"),
    (b"\xff\xfe\x00bad", "ecg.csv", 400, "INVALID_ECG_INPUT"),
])
def test_invalid_inputs_are_rejected(client, content, name, status, code):
    r = upload(client, content, name)
    assert r.status_code == status
    body = r.json()["error"]
    assert body["code"] == code and body["request_id"] and "Traceback" not in body["message"]


@needs_model
def test_payload_too_large(client):
    r = upload(client, b"x" * (2 * 1024 * 1024 + 10))
    assert r.status_code == 413 and r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


@needs_model
def test_case_insensitive_leads_and_100hz(client):
    upper = [n.upper() for n in LEADS]
    r = upload(client, csv_bytes(rows=1000, header=upper))
    assert r.status_code == 200
    body = r.json()
    assert body["input"]["sampling_rate_hz"] == 100 and body["input"]["samples"] == 1000


@needs_model
@pytest.mark.parametrize("sample", json.loads((DEMO_DIR / "manifest.json").read_text())["samples"]
                         if (DEMO_DIR / "manifest.json").is_file() else [])
def test_demo_samples_match_official_evaluation(client, sample):
    """Live CPU/FP32 inference on each demo ECG reproduces the frozen evaluation's recorded
    probabilities (GPU/bf16) for the same record, up to precision differences."""
    content = (REPO_ROOT / "frontend" / "public" / sample["file"].lstrip("/")).read_bytes()
    r = upload(client, content, Path(sample["file"]).name)
    assert r.status_code == 200
    body = r.json()
    assert body["input"] == {"format": "csv", "leads": 12, "samples": 5000,
                             "sampling_rate_hz": 500, "model_sampling_rate_hz": 100}
    assert len(body["waveform"]["data"]) == 12 and len(body["waveform"]["data"][0]) == 1000
    assert [p["label"] for p in body["predictions"]] == list(EXPECTED_THRESHOLDS)
    with (FROZEN_RUN / "test_predictions.csv").open() as f:
        row = next(r for r in csv.DictReader(f) if int(r["ecg_id"]) == sample["ecg_id"])
    for p in body["predictions"]:
        assert p["threshold"] == EXPECTED_THRESHOLDS[p["label"]]
        assert p["positive"] == (p["probability"] >= p["threshold"])
        assert abs(p["probability"] - float(row[f"prob_{p['label']}"])) < 0.02
    again = upload(client, content, Path(sample["file"]).name).json()   # deterministic
    assert [p["probability"] for p in again["predictions"]] == \
        [p["probability"] for p in body["predictions"]]
