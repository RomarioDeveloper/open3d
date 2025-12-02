# Food Weight Estimation Service

Микросервис для определения веса продуктов из изображений с YOLO сегментацией.

## Структура проекта

```
open3d/
├── images/          # Входные изображения
├── labels/          # YOLO segmentation labels
├── results/         # JSON результаты
├── food_weight.py   # Основной сервис
└── README.md
```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### Обработка папки с изображениями

```bash
python food_weight.py --input ./images --labels ./labels --output ./results
```

### Параметры

- `--input, -i`: Папка с изображениями (по умолчанию: `./images`)
- `--labels, -l`: Папка с YOLO labels (по умолчанию: `./labels`)
- `--output, -o`: Папка для результатов (по умолчанию: `./results`)
- `--plate-diameter, -d`: Диаметр тарелки в см (по умолчанию: 24.0)
- `--visualize, -v`: Показать 3D визуализацию моделей

## Формат результатов

Для каждого изображения создается JSON файл:

```json
{
  "image": "1.jpg",
  "objects": [
    {
      "image_path": "images/1.jpg",
      "object_id": 0,
      "class_id": 1,
      "food_type": "rice",
      "volume_cm3": 213.85,
      "weight_g": 181.77,
      "weight_kg": 0.1818,
      "density_g_cm3": 0.85,
      "polygon_points": 590,
      "area_pixels": 92298,
      "center_x": 0.23,
      "center_y": 0.552
    }
  ],
  "total_weight_g": 181.77,
  "total_objects": 1
}
```

Также создается `summary.json` со всеми результатами.

## Автоматическое определение типа продукта

Сервис автоматически определяет тип продукта по:
- Позиции на изображении (левая/правая сторона)
- Цвету сегментации
- Размеру и форме объекта

Поддерживаемые типы: rice, cabbage, potato, carrot, tomato, chicken, beef, pork, fish, pasta

