"""
Endpoints для работы с типами продуктов
"""

from fastapi import APIRouter
from app.models.food_types import FOOD_DENSITY, FOOD_HEIGHT_MAP

router = APIRouter()


@router.get("/food-types")
async def get_food_types():
    """
    Возвращает список всех поддерживаемых типов продуктов и их плотности
    """
    food_types = {}
    
    for food_type, density in FOOD_DENSITY.items():
        if food_type != 'default':
            food_types[food_type] = {
                "density_g_cm3": density,
                "default_height_cm": FOOD_HEIGHT_MAP.get(food_type, 3.0)
            }
    
    return {"food_types": food_types}

