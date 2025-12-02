"""
FastAPI REST API Server for Food Weight Estimation Service
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
import tempfile
import os
from pathlib import Path
import time
import logging

from food_weight import FoodWeightEstimator, FoodResult, FOOD_DENSITY
from dataclasses import asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Food Weight Estimation API",
    description="API для определения веса продуктов из изображений с YOLO сегментацией",
    version="1.0.0"
)

# CORS middleware для работы с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный экземпляр estimator
estimator = FoodWeightEstimator(plate_diameter_cm=24.0)


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Food Weight Estimation API"
    }


@app.get("/api/v1/health")
async def health_check_v1():
    """Проверка работоспособности сервиса (v1)"""
    return await health_check()


@app.get("/api/v1/food-types")
async def get_food_types():
    """Возвращает список всех поддерживаемых типов продуктов и их плотности"""
    height_map = {
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
    
    food_types = {}
    for food_type, density in FOOD_DENSITY.items():
        if food_type != 'default':
            food_types[food_type] = {
                "density_g_cm3": density,
                "default_height_cm": height_map.get(food_type, 3.0)
            }
    
    return {"food_types": food_types}


@app.post("/api/v1/process/image")
async def process_image(
    image: UploadFile = File(..., description="Изображение (JPG, PNG, BMP)"),
    labels: UploadFile = File(..., description="YOLO segmentation label файл (.txt)"),
    plate_diameter_cm: Optional[float] = Form(24.0, description="Диаметр тарелки в см")
):
    """
    Обрабатывает одно изображение с YOLO сегментацией и возвращает результаты определения веса продуктов
    """
    start_time = time.time()
    
    # Валидация файлов
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image file is required")
    
    if not labels.filename:
        raise HTTPException(status_code=400, detail="Labels file is required")
    
    # Проверка расширения изображения
    image_ext = Path(image.filename).suffix.lower()
    if image_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported image format: {image_ext}. Supported: .jpg, .jpeg, .png, .bmp"
        )
    
    # Создаем временные файлы
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Сохраняем изображение
        image_path = temp_path / image.filename
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # Сохраняем labels
        labels_path = temp_path / labels.filename
        with open(labels_path, "wb") as f:
            content = await labels.read()
            f.write(content)
        
        try:
            # Устанавливаем диаметр тарелки
            estimator.plate_diameter_cm = plate_diameter_cm
            
            # Обрабатываем изображение
            results = estimator.process_image(str(image_path), str(labels_path), visualize=False)
            
            if not results:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "image_name": image.filename,
                        "objects": [],
                        "total_weight_g": 0.0,
                        "total_objects": 0,
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                        "message": "No food objects detected"
                    }
                )
            
            # Формируем ответ
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": True,
                "image_name": image.filename,
                "objects": [asdict(r) for r in results],
                "total_weight_g": round(sum(r.weight_g for r in results), 2),
                "total_objects": len(results),
                "processing_time_ms": processing_time_ms
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process image: {str(e)}"
            )


@app.post("/api/v1/process/batch")
async def process_batch(
    images: List[UploadFile] = File(..., description="Массив изображений"),
    labels: List[UploadFile] = File(..., description="Массив YOLO label файлов"),
    plate_diameter_cm: Optional[float] = Form(24.0, description="Диаметр тарелки в см")
):
    """
    Обрабатывает несколько изображений одновременно
    """
    start_time = time.time()
    
    if len(images) != len(labels):
        raise HTTPException(
            status_code=400,
            detail=f"Number of images ({len(images)}) must match number of labels ({len(labels)})"
        )
    
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    estimator.plate_diameter_cm = plate_diameter_cm
    
    all_results = {}
    total_objects = 0
    total_weight = 0.0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for image_file, labels_file in zip(images, labels):
            try:
                # Сохраняем файлы
                image_path = temp_path / image_file.filename
                with open(image_path, "wb") as f:
                    f.write(await image_file.read())
                
                labels_path = temp_path / labels_file.filename
                with open(labels_path, "wb") as f:
                    f.write(await labels_file.read())
                
                # Обрабатываем
                results = estimator.process_image(str(image_path), str(labels_path), visualize=False)
                
                if results:
                    all_results[image_file.filename] = [asdict(r) for r in results]
                    total_objects += len(results)
                    total_weight += sum(r.weight_g for r in results)
                else:
                    all_results[image_file.filename] = []
                    
            except Exception as e:
                logger.error(f"Error processing {image_file.filename}: {e}", exc_info=True)
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


@app.get("/api/v1/process/status/{task_id}")
async def get_process_status(task_id: str):
    """
    Получает статус асинхронной задачи обработки
    В текущей реализации всегда возвращает, что задача не найдена,
    так как асинхронная обработка не реализована
    """
    return {
        "task_id": task_id,
        "status": "not_implemented",
        "message": "Async processing is not implemented in this version. Use /process/image or /process/batch endpoints directly."
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Food Weight Estimation API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Food Weight Estimation API server on http://{args.host}:{args.port}")
    logger.info(f"API documentation available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

