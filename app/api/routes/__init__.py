"""
API router package for CBL-04 backend.
"""

from fastapi import APIRouter
from app.api.routes.health import router as health_router
from app.api.routes.analyze import router as analyze_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(analyze_router)

__all__ = ["api_router"]

