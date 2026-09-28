"""Structured API errors: {"error": {"code", "message", "request_id"}}; no stack traces."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("cardiomamba.api")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _body(request: Request, code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message,
                      "request_id": getattr(request.state, "request_id", "")}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(_body(request, exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(_body(request, "INVALID_REQUEST",
                                  "Expected multipart/form-data with a 'file' field."),
                            status_code=422)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error (request %s)", getattr(request.state, "request_id", ""))
        return JSONResponse(_body(request, "INTERNAL_ERROR",
                                  "Inference failed due to an internal error."), status_code=500)
