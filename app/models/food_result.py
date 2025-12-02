"""
Модель результата обработки продукта
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FoodResult:
    """Результат обработки одного объекта продукта"""
    
    image_path: str
    object_id: int
    class_id: int
    food_type: str
    volume_cm3: float
    weight_g: float
    weight_kg: float
    density_g_cm3: float
    polygon_points: int
    area_pixels: int
    center_x: float
    center_y: float
    
    def to_dict(self) -> dict:
        """Преобразовать в словарь"""
        return {
            "image_path": self.image_path,
            "object_id": self.object_id,
            "class_id": self.class_id,
            "food_type": self.food_type,
            "volume_cm3": round(self.volume_cm3, 2),
            "weight_g": round(self.weight_g, 2),
            "weight_kg": round(self.weight_kg, 4),
            "density_g_cm3": self.density_g_cm3,
            "polygon_points": self.polygon_points,
            "area_pixels": self.area_pixels,
            "center_x": round(self.center_x, 3),
            "center_y": round(self.center_y, 3),
        }

