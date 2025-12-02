"""
Скрипт для обработки директории с изображениями (локальное использование)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.food_weight_service import FoodWeightService
from app.core.logging_config import setup_logging

# Настройка логирования
setup_logging()
logger = logging.getLogger(__name__)


def process_directory(
    input_dir: str = './images',
    labels_dir: str = './labels',
    output_dir: str = './results',
    plate_diameter_cm: float = 24.0,
    visualize: bool = False
):
    """
    Обработка всех изображений в директории
    
    Args:
        input_dir: Директория с изображениями
        labels_dir: Директория с YOLO label файлами
        output_dir: Директория для JSON результатов
        plate_diameter_cm: Диаметр тарелки для калибровки
        visualize: Показывать ли 3D визуализацию для каждого изображения
    """
    input_path = Path(input_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"Директория с изображениями не найдена: {input_dir}")
        return
    
    if not labels_path.exists():
        logger.error(f"Директория с labels не найдена: {labels_dir}")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"Изображения не найдены в {input_dir}")
        return
    
    logger.info(f"Найдено {len(image_files)} изображений")
    
    estimator = FoodWeightService(plate_diameter_cm=plate_diameter_cm)
    all_results = {}
    
    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            logger.warning(f"Файл labels не найден для {img_file.name}")
            continue
        
        try:
            results = estimator.process_image(str(img_file), str(label_file), visualize=visualize)
            
            if results:
                all_results[img_file.name] = [r.to_dict() for r in results]
                
                # Сохранение индивидуального JSON
                output_file = output_path / f"{img_file.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'image': str(img_file.name),
                        'objects': [r.to_dict() for r in results],
                        'total_weight_g': sum(r.weight_g for r in results),
                        'total_objects': len(results)
                    }, f, indent=2, ensure_ascii=False)
                
                logger.info(
                    f"Обработано {img_file.name}: {len(results)} объектов, "
                    f"общий вес: {sum(r.weight_g for r in results):.1f}g"
                )
        
        except Exception as e:
            logger.error(f"Ошибка обработки {img_file.name}: {e}")
    
    # Сохранение summary
    summary_file = output_path / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_images': len(all_results),
            'total_objects': sum(len(objs) for objs in all_results.values()),
            'total_weight_g': sum(
                sum(obj['weight_g'] for obj in objs)
                for objs in all_results.values()
            ),
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Результаты сохранены в {output_path}")
    logger.info(
        f"Итого: {len(all_results)} изображений, "
        f"{sum(len(objs) for objs in all_results.values())} объектов"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Food Weight Estimation Service')
    parser.add_argument('--input', '-i', default='./images', help='Директория с изображениями')
    parser.add_argument('--labels', '-l', default='./labels', help='Директория с labels')
    parser.add_argument('--output', '-o', default='./results', help='Директория для результатов')
    parser.add_argument('--plate-diameter', '-d', type=float, default=24.0,
                       help='Диаметр тарелки в см')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Показать 3D визуализацию для каждого изображения')
    
    args = parser.parse_args()
    
    process_directory(
        input_dir=args.input,
        labels_dir=args.labels,
        output_dir=args.output,
        plate_diameter_cm=args.plate_diameter,
        visualize=args.visualize
    )

