"""
Парсер YOLO segmentation формата
"""

import numpy as np
from typing import List, Dict
from pathlib import Path
from app.models.food_types import CLASS_NAMES
from app.exceptions import LabelParsingError


def parse_yolo_segmentation(
    label_path: str,
    img_width: int,
    img_height: int
) -> List[Dict]:
    """
    Парсинг YOLO segmentation формата
    
    Args:
        label_path: Путь к файлу labels
        img_width: Ширина изображения в пикселях
        img_height: Высота изображения в пикселях
        
    Returns:
        Список объектов с информацией о сегментации
        
    Raises:
        LabelParsingError: Если не удалось распарсить файл
    """
    if not Path(label_path).exists():
        raise LabelParsingError(f"Файл labels не найден: {label_path}")
    
    objects = []
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 7:
                    continue
                
                try:
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:-1]))  # Пропускаем confidence score
                    
                    points = []
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            x = int(coords[i] * img_width)
                            y = int(coords[i + 1] * img_height)
                            points.append([x, y])
                    
                    if len(points) >= 3:
                        objects.append({
                            'class_id': class_id,
                            'class_name': CLASS_NAMES.get(class_id, f'class_{class_id}'),
                            'polygon': np.array(points, dtype=np.int32)
                        })
                
                except (ValueError, IndexError) as e:
                    raise LabelParsingError(
                        f"Ошибка парсинга строки {line_num} в файле {label_path}: {e}"
                    )
    
    except IOError as e:
        raise LabelParsingError(f"Ошибка чтения файла {label_path}: {e}")
    
    return objects

