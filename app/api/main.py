"""
Главный файл FastAPI приложения
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.api.v1 import router as v1_router
from app.api.exceptions import setup_exception_handlers

# Настройка логирования
setup_logging()

# Получение настроек
settings = get_settings()

# Создание приложения
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Настройка обработчиков исключений
setup_exception_handlers(app)

# Подключение роутеров
app.include_router(v1_router, prefix="/api/v1", tags=["v1"])

# Health check endpoint (без версии)
@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "service": settings.API_TITLE
    }

