"""
Endpoints для обработки изображений
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import tempfile
import time
import logging
from pathlib import Path

from app.services.food_weight_service import FoodWeightService
from app.core.config import get_settings
from app.utils.file_validators import validate_image_file, validate_label_file
from app.exceptions import FoodWeightException

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# Глобальный экземпляр сервиса
food_weight_service = FoodWeightService(plate_diameter_cm=settings.DEFAULT_PLATE_DIAMETER_CM)


@router.post("/process/image")
async def process_image(
    image: UploadFile = File(..., description="Изображение (JPG, PNG, BMP)"),
    labels: UploadFile = File(..., description="YOLO segmentation label файл (.txt)"),
    plate_diameter_cm: Optional[float] = Form(None, description="Диаметр тарелки в см")
):
    """
    Обрабатывает одно изображение с YOLO сегментацией и возвращает результаты определения веса продуктов
    """
    start_time = time.time()
    
    try:
        # Валидация файлов
        validate_image_file(image)
        validate_label_file(labels)
        
        # Установка диаметра тарелки
        if plate_diameter_cm is not None:
            food_weight_service.plate_diameter_cm = plate_diameter_cm
        
        # Создание временных файлов
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Сохранение изображения
            image_path = temp_path / image.filename
            with open(image_path, "wb") as f:
                content = await image.read()
                f.write(content)
            
            # Сохранение labels
            labels_path = temp_path / labels.filename
            with open(labels_path, "wb") as f:
                content = await labels.read()
                f.write(content)
            
            # Обработка изображения
            results = food_weight_service.process_image(
                str(image_path),
                str(labels_path),
                visualize=False
            )
            
            if not results:
                processing_time_ms = int((time.time() - start_time) * 1000)
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "image_name": image.filename,
                        "objects": [],
                        "total_weight_g": 0.0,
                        "total_objects": 0,
                        "processing_time_ms": processing_time_ms,
                        "message": "Объекты продуктов не обнаружены"
                    }
                )
            
            # Формирование ответа
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": True,
                "image_name": image.filename,
                "objects": [r.to_dict() for r in results],
                "total_weight_g": round(sum(r.weight_g for r in results), 2),
                "total_objects": len(results),
                "processing_time_ms": processing_time_ms
            }
    
    except FoodWeightException as e:
        logger.error(f"Ошибка обработки изображения: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Неожиданная ошибка при обработке изображения: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки изображения: {str(e)}"
        )


@router.post("/process/batch")
async def process_batch(
    images: List[UploadFile] = File(..., description="Массив изображений"),
    labels: List[UploadFile] = File(..., description="Массив YOLO label файлов"),
    plate_diameter_cm: Optional[float] = Form(None, description="Диаметр тарелки в см")
):
    """
    Обрабатывает несколько изображений одновременно
    """
    start_time = time.time()
    
    if len(images) != len(labels):
        raise HTTPException(
            status_code=400,
            detail=f"Количество изображений ({len(images)}) должно совпадать с количеством labels ({len(labels)})"
        )
    
    if not images:
        raise HTTPException(status_code=400, detail="Требуется хотя бы одно изображение")
    
    # Установка диаметра тарелки
    if plate_diameter_cm is not None:
        food_weight_service.plate_diameter_cm = plate_diameter_cm
    
    all_results = {}
    total_objects = 0
    total_weight = 0.0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for image_file, labels_file in zip(images, labels):
            try:
                # Валидация файлов
                validate_image_file(image_file)
                validate_label_file(labels_file)
                
                # Сохранение файлов
                image_path = temp_path / image_file.filename
                with open(image_path, "wb") as f:
                    f.write(await image_file.read())
                
                labels_path = temp_path / labels_file.filename
                with open(labels_path, "wb") as f:
                    f.write(await labels_file.read())
                
                # Обработка
                results = food_weight_service.process_image(
                    str(image_path),
                    str(labels_path),
                    visualize=False
                )
                
                if results:
                    all_results[image_file.filename] = [r.to_dict() for r in results]
                    total_objects += len(results)
                    total_weight += sum(r.weight_g for r in results)
                else:
                    all_results[image_file.filename] = []
            
            except FoodWeightException as e:
                logger.error(f"Ошибка обработки {image_file.filename}: {e}", exc_info=True)
                all_results[image_file.filename] = []
            
            except Exception as e:
                logger.error(f"Неожиданная ошибка при обработке {image_file.filename}: {e}", exc_info=True)
                all_results[image_file.filename] = []
    
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    return {
        "success": True,
        "total_images": len(images),
        "total_objects": total_objects,
        "total_weight_g": round(total_weight, 2),
        "results": all_results,
        "processing_time_ms": processing_time_ms
    }


@router.get("/process/status/{task_id}")
async def get_process_status(task_id: str):
    """
    Получает статус асинхронной задачи обработки
    В текущей реализации всегда возвращает, что задача не найдена,
    так как асинхронная обработка не реализована
    """
    return {
        "task_id": task_id,
        "status": "not_implemented",
        "message": "Асинхронная обработка не реализована в этой версии. Используйте /process/image или /process/batch endpoints напрямую."
    }

