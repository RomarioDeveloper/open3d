"""
Обработчики исключений для API
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

from app.exceptions import FoodWeightException

logger = logging.getLogger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """Настройка обработчиков исключений"""
    
    @app.exception_handler(FoodWeightException)
    async def food_weight_exception_handler(request: Request, exc: FoodWeightException):
        """Обработчик кастомных исключений"""
        logger.error(f"FoodWeightException: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": exc.__class__.__name__,
                "message": str(exc),
                "detail": "Ошибка обработки данных о продуктах"
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Обработчик ошибок валидации"""
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "ValidationError",
                "message": "Ошибка валидации входных данных",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Обработчик общих исключений"""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "Внутренняя ошибка сервера",
                "detail": str(exc) if logger.level <= logging.DEBUG else "Обратитесь к администратору"
            }
        )

