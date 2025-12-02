"""
API v1 endpoints
"""

from fastapi import APIRouter
from app.api.v1 import health, food_types, process

router = APIRouter()

# Подключение роутеров
router.include_router(health.router, tags=["health"])
router.include_router(food_types.router, tags=["food-types"])
router.include_router(process.router, tags=["process"])

