"""
Health check endpoints
"""

from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "service": settings.API_TITLE
    }

