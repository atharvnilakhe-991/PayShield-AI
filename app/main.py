"""
FastAPI Main Application Entry Point for CBL-04 Google Pay Screenshot Manipulation Detection.
Configures CORS middleware, registers health and analysis endpoints,
and implements global exception handlers adhering to Phase 1 APIErrorResponse contracts.
"""

import logging
import uuid
from typing import Any
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import api_router
from app.core.config import get_settings
from app.schemas.enums import ErrorCodeEnum
from app.schemas.errors import create_error_response, sanitize_error_detail
from app.services import (
    CNNModelError,
    CNNModelUnavailableError,
    OCRError,
    OCRImageError,
    OCRInitializationError,
)
from app.services.image_validation_service import ImageValidationError

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cbl04.api")


def create_application() -> FastAPI:
    """Factory creating and configuring the FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 1. CORS Middleware Configuration
    origins = settings.CORS_ORIGINS
    if isinstance(origins, str):
        origins = [origins]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Request Tracing Middleware
    @app.middleware("http")
    async def trace_id_middleware(request: Request, call_next: Any):
        trace_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("x-request-id")
            or f"req-{uuid.uuid4().hex[:12]}"
        )
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = trace_id
        return response

    # 3. Global Exception Handlers adhering to APIErrorResponse

    @app.exception_handler(ImageValidationError)
    async def image_validation_exception_handler(
        request: Request, exc: ImageValidationError
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None)
        logger.warning(
            "Image validation error [Trace: %s, Code: %s]: %s",
            trace_id,
            exc.error_code,
            exc.message,
        )
        error_resp = create_error_response(
            error_code=exc.error_code,
            message=exc.message,
            status_code=exc.status_code,
            request_id=trace_id,
            detail=exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(CNNModelUnavailableError)
    async def cnn_model_unavailable_handler(
        request: Request, exc: CNNModelUnavailableError
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None)
        logger.error("CNN model unavailable [Trace: %s]: %s", trace_id, exc)
        error_resp = create_error_response(
            error_code=ErrorCodeEnum.MODEL_UNAVAILABLE,
            message="CNN model service is currently unavailable.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            request_id=trace_id,
            detail=None,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(CNNModelError)
    async def cnn_model_error_handler(
        request: Request, exc: CNNModelError
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None)
        logger.error("CNN model processing error [Trace: %s]: %s", trace_id, exc)
        error_resp = create_error_response(
            error_code=ErrorCodeEnum.INTERNAL_PROCESSING_FAILURE,
            message="An error occurred during CNN visual feature extraction.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=trace_id,
            detail=None,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(OCRInitializationError)
    async def ocr_initialization_error_handler(
        request: Request, exc: OCRInitializationError
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None)
        logger.error("OCR engine initialization error [Trace: %s]: %s", trace_id, exc)
        error_resp = create_error_response(
            error_code=ErrorCodeEnum.MODEL_UNAVAILABLE,
            message="OCR extraction service is currently unavailable.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            request_id=trace_id,
            detail=None,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(OCRImageError)
    async def ocr_image_error_handler(
        request: Request, exc: OCRImageError
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None)
        logger.error("OCR image decoding error [Trace: %s]: %s", trace_id, exc)
        error_resp = create_error_response(
            error_code=ErrorCodeEnum.IMAGE_CORRUPTED,
            message=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
            request_id=trace_id,
            detail=None,
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(OCRError)
    async def ocr_error_handler(
        request: Request, exc: OCRError
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None)
        logger.error("OCR processing error [Trace: %s]: %s", trace_id, exc)
        error_resp = create_error_response(
            error_code=ErrorCodeEnum.INTERNAL_PROCESSING_FAILURE,
            message="An error occurred during OCR text extraction.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=trace_id,
            detail=None,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None)
        errors = exc.errors()

        # Check specifically if the required 'image' file parameter was omitted
        is_missing_image = any(
            err.get("loc") and "image" in [str(loc) for loc in err["loc"]]
            for err in errors
        )

        if is_missing_image:
            error_code = ErrorCodeEnum.MISSING_REQUIRED_INPUT
            status_code = status.HTTP_400_BAD_REQUEST
            message = "Missing required form field 'image'. Please upload a Google Pay receipt screenshot."
        else:
            error_code = ErrorCodeEnum.VALIDATION_ERROR
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
            message = "Request payload validation failed."

        # Sanitize error detail list
        sanitized_details = [
            {
                "field": " -> ".join(str(loc) for loc in err.get("loc", [])),
                "message": err.get("msg"),
                "type": err.get("type"),
            }
            for err in errors
        ]

        logger.warning(
            "Request validation error [Trace: %s]: %s", trace_id, sanitized_details
        )
        error_resp = create_error_response(
            error_code=error_code,
            message=message,
            status_code=status_code,
            request_id=trace_id,
            detail=sanitized_details,
        )
        return JSONResponse(
            status_code=status_code,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None)
        status_code = exc.status_code

        # Map HTTP status to appropriate error code
        if status_code == 413:
            code = ErrorCodeEnum.FILE_TOO_LARGE
        elif status_code == 415:
            code = ErrorCodeEnum.INVALID_IMAGE_FORMAT
        elif status_code == 422:
            code = ErrorCodeEnum.VALIDATION_ERROR
        elif status_code >= 500:
            code = ErrorCodeEnum.INTERNAL_PROCESSING_FAILURE
        else:
            code = ErrorCodeEnum.VALIDATION_ERROR

        error_resp = create_error_response(
            error_code=code,
            message=str(exc.detail) if exc.detail else "HTTP error encountered.",
            status_code=status_code,
            request_id=trace_id,
        )
        return JSONResponse(
            status_code=status_code,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None)
        # Log stack trace on server only; never expose to client
        logger.exception("Unhandled server exception [Trace: %s]: %s", trace_id, exc)

        error_resp = create_error_response(
            error_code=ErrorCodeEnum.INTERNAL_PROCESSING_FAILURE,
            message="An internal server error occurred while processing the screenshot.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=trace_id,
            detail=None,  # Do not leak stack traces or internal paths
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_resp.model_dump(),
        )

    # 4. Include Routers
    app.include_router(api_router)

    return app


app = create_application()

