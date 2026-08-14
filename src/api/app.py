"""FastAPI application for registered Phase 11 forecast models."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.config import API_MAX_PAYLOAD_BYTES, APP_NAME, APP_VERSION
from src.forecasting.logging_utils import configure_logging

logger = logging.getLogger("forecast_service.api")


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(
        title=f"{APP_NAME} Forecast API",
        version=APP_VERSION,
        description=(
            "Academic/reference forecast API over Phase 11 registered models. "
            "Authentication is not included in this academic/reference implementation."
        ),
    )
    application.state.engines = {}

    @application.middleware("http")
    async def limit_payload(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > API_MAX_PAYLOAD_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Payload exceeds {API_MAX_PAYLOAD_BYTES} bytes"},
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
        return await call_next(request)

    @application.exception_handler(Exception)
    async def unhandled(_, exc: Exception):
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        logger.error("unhandled_error type=%s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @application.exception_handler(RequestValidationError)
    async def req_validation(_, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": "Invalid request schema"})

    application.include_router(router)
    return application


app = create_app()
