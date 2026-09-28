"""CardioMamba inference API (research/educational demo; inference only, no training).

Run from backend/:  uv run uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.errors import install_error_handlers
from app.ml.model_loader import ModelLoadError, load_runtime

logger = logging.getLogger("cardiomamba.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.runtime = None
        try:                                    # load the frozen model once at startup
            app.state.runtime = load_runtime(settings.checkpoint_path, settings.thresholds_path,
                                             settings.test_metrics_path, settings.repo_root,
                                             settings.device)
            logger.info("Model loaded on %s", app.state.runtime.device)
        except (ModelLoadError, OSError, KeyError, ValueError) as exc:
            app.state.load_error = str(exc)
            logger.error("Model not loaded: %s", exc)
        yield

    app = FastAPI(title="CardioMamba API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    install_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
