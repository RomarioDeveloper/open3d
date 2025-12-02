"""
Точка входа для запуска API сервера
"""

import uvicorn
import argparse
import logging
from app.core.config import get_settings
from app.core.logging_config import setup_logging

# Настройка логирования
setup_logging()
logger = logging.getLogger(__name__)

# Получение настроек
settings = get_settings()


def main():
    """Главная функция для запуска сервера"""
    parser = argparse.ArgumentParser(description='Food Weight Estimation API Server')
    parser.add_argument('--host', default=settings.HOST, help=f'Host to bind (default: {settings.HOST})')
    parser.add_argument('--port', type=int, default=settings.PORT, help=f'Port to bind (default: {settings.PORT})')
    parser.add_argument('--reload', action='store_true', default=settings.RELOAD,
                       help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    logger.info(f"Запуск Food Weight Estimation API сервера на http://{args.host}:{args.port}")
    logger.info(f"API документация доступна на http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "app.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()

