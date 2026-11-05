"""
Health check endpoints for CBL-04 backend service.
Provides non-blocking service liveness verification.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import get_settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response payload."""
    status: str = Field(default="healthy", description="Service health state")
    service: str = Field(default="cbl04-gpay-detection-api", description="Service identifier")
    version: str = Field(default="0.1.0", description="Service application version")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns backend liveness status and version metadata without invoking ML models.",
)
async def get_health() -> HealthResponse:
    """Returns 200 OK with health status."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        service=settings.SERVICE_NAME,
        version=settings.API_VERSION,
    )


@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    include_in_schema=False,
)
async def get_health_v1() -> HealthResponse:
    """Compatibility alias for /health under /api/v1 prefix."""
    return await get_health()

