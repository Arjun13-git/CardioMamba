"""FastAPI application placeholder. Implement during Phase 7."""
from fastapi import FastAPI

app = FastAPI(title="CardioMamba API", version="0.1.0")

@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok", "service": "cardiomamba-api"}
