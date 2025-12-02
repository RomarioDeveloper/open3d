"""
Пример использования API для анализа фото
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь (если нужно)
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import requests
import json

# Настройки
API_URL = "http://localhost:8000/api/v1/process/image"
IMAGES_DIR = Path("images")
LABELS_DIR = Path("labels")


def process_single_image(image_name: str, plate_diameter: float = 24.0):
    """
    Обработка одного изображения через API
    
    Args:
        image_name: Имя изображения (например, "1.jpg")
        plate_diameter: Диаметр тарелки в см
    """
    # Пути к файлам
    image_path = IMAGES_DIR / image_name
    label_path = LABELS_DIR / f"{Path(image_name).stem}.txt"
    
    if not image_path.exists():
        print(f"❌ Изображение не найдено: {image_path}")
        return
    
    if not label_path.exists():
        print(f"❌ Файл labels не найден: {label_path}")
        return
    
    print(f"📸 Обработка: {image_name}")
    
    # Подготовка файлов
    files = {
        'image': (image_name, open(image_path, 'rb'), 'image/jpeg'),
        'labels': (label_path.name, open(label_path, 'rb'), 'text/plain')
    }
    
    # Параметры
    data = {
        'plate_diameter_cm': plate_diameter
    }
    
    try:
        # Отправка запроса
        response = requests.post(API_URL, files=files, data=data)
        
        # Закрытие файлов
        files['image'][1].close()
        files['labels'][1].close()
        
        # Проверка ответа
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Успешно обработано!")
            print(f"   Объектов: {result['total_objects']}")
            print(f"   Общий вес: {result['total_weight_g']} г")
            print(f"   Время обработки: {result['processing_time_ms']} мс")
            
            # Детали по каждому объекту
            if result['objects']:
                print("\n   Детали:")
                for obj in result['objects']:
                    print(f"   - {obj['food_type']}: {obj['weight_g']} г "
                          f"(объем: {obj['volume_cm3']} см³)")
            
            return result
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"   {response.text}")
            return None
    
    except requests.exceptions.ConnectionError:
        print("❌ Не удалось подключиться к API серверу")
        print("   Убедитесь, что сервер запущен: python main.py")
        return None
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def process_batch(image_names: list, plate_diameter: float = 24.0):
    """
    Пакетная обработка нескольких изображений
    
    Args:
        image_names: Список имен изображений
        plate_diameter: Диаметр тарелки в см
    """
    print(f"📦 Пакетная обработка {len(image_names)} изображений...")
    
    # Подготовка файлов
    files = {}
    for image_name in image_names:
        image_path = IMAGES_DIR / image_name
        label_path = LABELS_DIR / f"{Path(image_name).stem}.txt"
        
        if image_path.exists() and label_path.exists():
            files[f'images'] = (image_name, open(image_path, 'rb'), 'image/jpeg')
            files[f'labels'] = (label_path.name, open(label_path, 'rb'), 'text/plain')
        else:
            print(f"⚠️  Пропущено: {image_name} (файлы не найдены)")
    
    if not files:
        print("❌ Нет файлов для обработки")
        return
    
    data = {'plate_diameter_cm': plate_diameter}
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/process/batch",
            files=files,
            data=data
        )
        
        # Закрытие файлов
        for file_obj in files.values():
            file_obj[1].close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Обработано изображений: {result['total_images']}")
            print(f"   Всего объектов: {result['total_objects']}")
            print(f"   Общий вес: {result['total_weight_g']} г")
            return result
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(response.text)
            return None
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


if __name__ == "__main__":
    print("=" * 50)
    print("Тестирование API для анализа фото")
    print("=" * 50)
    print()
    
    # Проверка доступности сервера
    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        if health.status_code == 200:
            print("✅ API сервер доступен")
        else:
            print("⚠️  API сервер отвечает, но с ошибкой")
    except:
        print("❌ API сервер недоступен!")
        print("   Запустите сервер: python main.py")
        exit(1)
    
    print()
    
    # Пример 1: Обработка одного изображения
    print("Пример 1: Обработка одного изображения")
    print("-" * 50)
    result = process_single_image("1.jpg", plate_diameter=24.0)
    
    print()
    print()
    
    # Пример 2: Пакетная обработка
    print("Пример 2: Пакетная обработка")
    print("-" * 50)
    images = ["1.jpg", "22.jpg", "81.jpg", "4323.jpg"]
    process_batch(images, plate_diameter=24.0)
    
    print()
    print("=" * 50)
    print("Готово!")

