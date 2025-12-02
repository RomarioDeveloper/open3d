"""
Константы и типы для продуктов
"""

from typing import Dict

# База данных плотности продуктов (g/cm³)
FOOD_DENSITY: Dict[str, float] = {
    'rice': 0.85,
    'cabbage': 0.55,
    'potato': 0.75,
    'carrot': 0.64,
    'tomato': 0.60,
    'chicken': 1.05,
    'beef': 1.06,
    'pork': 1.04,
    'fish': 1.05,
    'pasta': 0.70,
    'default': 0.70,
}

# Карта высот по умолчанию для разных типов продуктов (см)
FOOD_HEIGHT_MAP: Dict[str, float] = {
    'rice': 2.0,
    'cabbage': 4.0,
    'potato': 3.0,
    'carrot': 2.5,
    'tomato': 3.5,
    'chicken': 2.5,
    'beef': 2.5,
    'pork': 2.5,
    'fish': 2.0,
    'pasta': 2.0,
    'default': 3.0,
}

# Имена классов YOLO
CLASS_NAMES: Dict[int, str] = {
    0: 'plate',
    1: 'food',
    2: 'glass'
}

# Тип продукта (для валидации)
FoodType = str

# Плотность продукта
FoodDensity = float

