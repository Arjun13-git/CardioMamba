# CardioMamba — Demo Frontend

Next.js (App Router) + TypeScript + Tailwind research-demo UI for the frozen CardioMamba model.
Research/educational prototype — predictions are model outputs, not medical diagnoses.

## Run

```bash
# 1) inference backend (from backend/)
uv sync && uv run uvicorn app.main:app --port 8000
# 2) frontend (from frontend/)
npm install && npm run dev        # http://localhost:3000
```

`/api/v1/*` is proxied to the backend (`CARDIOMAMBA_API_URL`, default `http://127.0.0.1:8000`).

## Pages

- `/` — dashboard: project summary, key facts, architecture pipeline
- `/analyze` — load a bundled demo ECG or upload a CSV; 12-lead viewer; live predictions with
  frozen validation thresholds
- `/architecture` — tensor flow, bidirectional block, S6 mixer, model specification
- `/results` — frozen held-out (fold 10) metrics, per-class table, training curves

## Data provenance

`data/*.json` and `public/demo/*` are generated from the frozen run
(`outputs/cardiomamba/full-30ep-seed42/`) by `scripts/export_demo_assets.py` (read-only). Demo
ECGs are PTB-XL 1.0.3 fold-10 records (CC BY 4.0), selected by a fixed rule independent of
model predictions; they are not newly collected ECGs.
