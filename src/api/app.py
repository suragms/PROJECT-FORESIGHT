"""FastAPI application for registered Phase 11 forecast models."""

from __future__ import annotations

import os
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.metrics import incr, observe_latency
from src.api.routes import router
from src.config import (
    APP_NAME,
    APP_VERSION,
    api_max_payload_bytes,
    foresight_env,
    rate_limit_enabled,
    rate_limit_forecast_requests,
    rate_limit_requests,
    rate_limit_window_seconds,
)
from src.forecasting.logging_utils import configure_logging
from src.production.config_validation import ConfigValidationError, assert_runtime_config, config_snapshot
from src.security.audit import audit
from src.security.auth import auth_is_required, extract_api_key, is_public_path, key_matches
from src.security.rate_limit import RateLimiter, client_key

logger = logging.getLogger("forecast_service.api")

FORECAST_PATHS = frozenset({"/forecast", "/forecast/batch"})
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "Cache-Control": "no-store",
}


@asynccontextmanager
async def lifespan(application: FastAPI):
    configure_logging()
    env = foresight_env()
    try:
        assert_runtime_config()
    except ConfigValidationError as exc:
        audit("application_startup_notice", environment=env, error=str(exc))
        logger.warning("Config validation notice: %s", exc)
    from src.production.config_validation import validate_runtime_config

    warnings = validate_runtime_config()
    if warnings:
        logger.warning("config_warnings %s", warnings)
    audit(
        "application_startup",
        environment=env,
        version=APP_VERSION,
        config=config_snapshot(),
    )
    try:
        from src.auth.database import init_db
        init_db()
    except Exception as exc:
        logger.warning("Auth DB initialization fallback notice: %s", exc)
    yield
    audit("application_shutdown", environment=env, version=APP_VERSION)


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(
        title=f"{APP_NAME} Forecast API",
        version=APP_VERSION,
        description=(
            "Academic/reference forecast API over Phase 11 registered models. "
            "Authentication and rate limiting are configurable through environment variables."
        ),
        lifespan=lifespan,
    )
    application.state.engines = {}
    application.state.rate_limiter = RateLimiter()

    @application.middleware("http")
    async def limit_payload(request: Request, call_next):
        cl = request.headers.get("content-length")
        limit = api_max_payload_bytes()
        if cl is not None:
            try:
                if int(cl) > limit:
                    audit("validation_failure", path=str(request.url.path), reason="payload_too_large")
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Payload exceeds {limit} bytes"},
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
        return await call_next(request)

    @application.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response

    @application.middleware("http")
    async def authenticate(request: Request, call_next):
        if is_public_path(request.url.path):
            return await call_next(request)
        if not auth_is_required():
            return await call_next(request)
        provided = extract_api_key(request)
        if key_matches(provided):
            if request.url.path in {"/model", "/forecast", "/forecast/batch"}:
                audit("authentication_success", path=request.url.path)
            return await call_next(request)
        incr("auth_failures")
        audit(
            "authentication_failure",
            path=request.url.path,
            reason="missing" if not provided else "invalid",
        )
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    @application.middleware("http")
    async def rate_limit(request: Request, call_next):
        if is_public_path(request.url.path) or not rate_limit_enabled():
            return await call_next(request)
        limiter: RateLimiter = request.app.state.rate_limiter
        host = request.client.host if request.client else "unknown"
        window = rate_limit_window_seconds()
        general_ok = limiter.allow(
            client_key(host, "general", "*"),
            rate_limit_requests(),
            window,
        )
        forecast_ok = True
        if request.url.path in FORECAST_PATHS and request.method == "POST":
            forecast_ok = limiter.allow(
                client_key(host, "forecast", request.url.path),
                rate_limit_forecast_requests(),
                window,
            )
        if not general_ok or not forecast_ok:
            incr("rate_limit_events")
            audit("rate_limit_rejection", path=request.url.path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(window)},
            )
        return await call_next(request)

    @application.middleware("http")
    async def metrics_mw(request: Request, call_next):
        incr("request_count")
        started = time.perf_counter()
        response = await call_next(request)
        observe_latency(time.perf_counter() - started)
        if response.status_code >= 400:
            incr("error_count")
        return response

    @application.middleware("http")
    async def request_id_mw(request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @application.exception_handler(Exception)
    async def unhandled(_, exc: Exception):
        if hasattr(exc, "status_code"):
            status = getattr(exc, "status_code", 500)
            detail = getattr(exc, "detail", str(exc))
            return JSONResponse(status_code=status, content={"detail": detail})
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        logger.error("unhandled_error type=%s", type(exc).__name__)
        audit("unhandled_error", error_type=type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @application.exception_handler(RequestValidationError)
    async def req_validation(_, exc: RequestValidationError):
        audit("validation_failure", reason="schema")
        return JSONResponse(status_code=422, content={"detail": "Invalid request schema"})

    application.include_router(router)
    from src.api.auth_routes import router as auth_router
    application.include_router(auth_router)
    from src.api.phase20_routes import router as phase20_router
    application.include_router(phase20_router, prefix="/phase20", tags=["phase20"])
    from src.api.phase21_routes import router as phase21_router
    application.include_router(phase21_router, prefix="/phase21", tags=["phase21"])
    return application


app = create_app()
