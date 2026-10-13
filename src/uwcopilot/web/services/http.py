"""FastAPI glue shared by the feature routers: error responses, the store dependency, no-store caching.

Error bodies keep FastAPI's shape, {"detail": "..."}, and add "reasons" when there are several problems:

    422 {"detail": "The checkpoint refused this decision.", "reasons": ["only a senior_underwriter may decline ..."]}
    503 {"detail": "The calculation took longer than 60 s. It keeps running on the server; try again in a minute."}
"""

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from uwcopilot.web.services.compute import ComputeBusy, ComputeTimeout
from uwcopilot.web.services.errors import ServiceError
from uwcopilot.web.store import DeletionPending, Store

BUSY_MESSAGE = "The server is busy with other calculations right now. Please try again in a moment."


def error_body(detail: str, reasons: list[str] | None = None) -> dict:
    """{"detail": ..., "reasons": [...]} (reasons only when there are any)."""
    body: dict = {"detail": detail}
    if reasons:
        body["reasons"] = reasons
    return body


async def service_error_handler(request: Request, error: Exception) -> JSONResponse:
    """A ServiceError as its HTTP status."""
    assert isinstance(error, ServiceError)
    return JSONResponse(error_body(error.message, error.reasons), status_code=error.status_code)


async def timeout_handler(request: Request, error: Exception) -> JSONResponse:
    """A heavy job that did not finish in time: 503 with what happens next."""
    message = (
        f"The calculation is taking longer than the server waits ({error}). It keeps running on the server; "
        "try again in a minute to get the result."
    )
    return JSONResponse(error_body(message), status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


async def busy_handler(request: Request, error: Exception) -> JSONResponse:
    """Too many heavy jobs at once: 503 "busy"."""
    return JSONResponse(error_body(BUSY_MESSAGE), status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


async def deletion_pending_handler(request: Request, error: Exception) -> JSONResponse:
    """A deletion whose remote copy could not be removed yet: 202 with an honest message."""
    message = (
        "Deleted here, but the copy in the backup dataset could not be deleted yet. The deletion is retried "
        "automatically; until then the data is not fully gone."
    )
    return JSONResponse(error_body(message), status_code=status.HTTP_202_ACCEPTED)


def add_error_handlers(app: FastAPI) -> None:
    """Register the handlers above on the app."""
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(ComputeTimeout, timeout_handler)
    app.add_exception_handler(ComputeBusy, busy_handler)
    app.add_exception_handler(DeletionPending, deletion_pending_handler)


def get_store(request: Request) -> Store:
    """FastAPI dependency: the app's Store (503 before startup has opened it)."""
    store: Store | None = request.app.state.store
    if store is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="storage is not ready")
    return store


def no_store(response: Response) -> None:
    """FastAPI dependency: personal data must not be kept in browser or proxy caches."""
    response.headers["Cache-Control"] = "no-store"
