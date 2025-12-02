"""
Сервис для определения веса продуктов
"""

import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from app.models.food_result import FoodResult
from app.models.food_types import FOOD_DENSITY, FOOD_HEIGHT_MAP
from app.utils.yolo_parser import parse_yolo_segmentation
from app.exceptions import (
    ImageProcessingError,
    CalibrationError,
    VolumeCalculationError,
)

logger = logging.getLogger(__name__)


class FoodWeightService:
    """Сервис для определения веса продуктов из изображений"""
    
    def __init__(self, plate_diameter_cm: float = 24.0):
        """
        Инициализация сервиса
        
        Args:
            plate_diameter_cm: Диаметр тарелки в см для калибровки
        """
        self.plate_diameter_cm = plate_diameter_cm
    
    def calibrate_from_plate(
        self,
        plate_polygon: np.ndarray,
        img_width: int,
        img_height: int,
        known_plate_diameter_cm: float
    ) -> float:
        """
        Калибровка масштаба по тарелке
        
        Args:
            plate_polygon: Полигон тарелки
            img_width: Ширина изображения
            img_height: Высота изображения
            known_plate_diameter_cm: Известный диаметр тарелки в см
            
        Returns:
            Пикселей на см
            
        Raises:
            CalibrationError: Если не удалось выполнить калибровку
        """
        try:
            bbox = cv2.boundingRect(plate_polygon)
            plate_width_pixels = bbox[2]
            plate_height_pixels = bbox[3]
            
            plate_diameter_pixels = (plate_width_pixels + plate_height_pixels) / 2.0
            
            if plate_diameter_pixels <= 0:
                raise CalibrationError("Некорректный размер тарелки для калибровки")
            
            pixels_per_cm = plate_diameter_pixels / known_plate_diameter_cm
            return pixels_per_cm
        
        except Exception as e:
            raise CalibrationError(f"Ошибка калибровки: {e}")
    
    def detect_food_type(
        self,
        polygon: np.ndarray,
        image: np.ndarray,
        mask: np.ndarray,
        img_width: int,
        img_height: int
    ) -> Tuple[str, float]:
        """
        Определение типа продукта по цвету, текстуре и форме
        
        Args:
            polygon: Полигон объекта
            image: Изображение
            mask: Маска объекта
            img_width: Ширина изображения
            img_height: Высота изображения
            
        Returns:
            Кортеж (тип продукта, высота в см)
        """
        h, w = image.shape[:2]
        
        # Извлечение замаскированной области
        masked_region = image[mask > 0]
        if len(masked_region) == 0:
            return 'default', FOOD_HEIGHT_MAP['default']
        
        # Конвертация в RGB если нужно
        if len(image.shape) == 3 and image.shape[2] == 3:
            masked_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[mask > 0]
        else:
            masked_rgb = masked_region
        
        # Анализ цвета
        avg_color = np.mean(masked_rgb, axis=0)
        std_color = np.std(masked_rgb, axis=0)
        avg_brightness = np.mean(avg_color)
        color_variance = np.mean(std_color)
        
        r, g, b = avg_color[0], avg_color[1], avg_color[2]
        
        # Характеристики формы
        area_pixels = np.sum(mask > 0)
        bbox = cv2.boundingRect(polygon)
        aspect_ratio = bbox[2] / max(bbox[3], 1)
        
        # Анализ текстуры
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        masked_gray = gray[mask > 0]
        texture_variance = np.var(masked_gray) if len(masked_gray) > 0 else 0
        
        # Доминирование цветов
        r_dominance = r / (r + g + b + 1e-6)
        g_dominance = g / (r + g + b + 1e-6)
        b_dominance = b / (r + g + b + 1e-6)
        
        # Система оценки для разных типов продуктов
        scores = {
            'rice': 0.0,
            'cabbage': 0.0,
            'potato': 0.0,
            'carrot': 0.0,
            'tomato': 0.0,
            'chicken': 0.0,
            'beef': 0.0,
            'pork': 0.0,
            'fish': 0.0,
            'pasta': 0.0,
        }
        
        # Рис: белый/бежевый, высокая яркость, низкая вариация цвета
        if avg_brightness > 140 and color_variance < 30:
            scores['rice'] += 3.0
        if avg_brightness > 150:
            scores['rice'] += 2.0
        if area_pixels < 60000 and aspect_ratio > 0.7:
            scores['rice'] += 1.0
        if r > 180 and g > 180 and b > 180:
            scores['rice'] += 2.0
        
        # Капуста: зеленая, средняя яркость, круглая форма
        if g_dominance > 0.35 and g > r and g > b:
            scores['cabbage'] += 3.0
        if 100 < avg_brightness < 160 and g > 100:
            scores['cabbage'] += 2.0
        if aspect_ratio > 0.7 and area_pixels > 30000:
            scores['cabbage'] += 1.0
        if g > 120 and r < 150 and b < 150:
            scores['cabbage'] += 2.0
        
        # Картофель: коричневый/желтый, средняя яркость
        if r_dominance > 0.35 and 100 < avg_brightness < 140:
            scores['potato'] += 2.0
        if 80 < r < 150 and 80 < g < 140 and 60 < b < 120:
            scores['potato'] += 2.0
        if aspect_ratio > 0.6 and area_pixels > 20000:
            scores['potato'] += 1.0
        
        # Морковь: оранжевая, высокое соотношение красного/зеленого
        if r > 150 and g > 100 and b < 100:
            scores['carrot'] += 3.0
        if r_dominance > 0.4 and g_dominance > 0.3:
            scores['carrot'] += 2.0
        if aspect_ratio < 0.6 or aspect_ratio > 1.5:
            scores['carrot'] += 1.0
        
        # Помидор: красный, высокая яркость, круглый
        if r > 150 and r > g * 1.5 and b < 100:
            scores['tomato'] += 3.0
        if r_dominance > 0.45:
            scores['tomato'] += 2.0
        if aspect_ratio > 0.7:
            scores['tomato'] += 1.0
        
        # Мясо (курица, говядина, свинина): коричневое/розовое
        if 60 < avg_brightness < 120 and texture_variance > 200:
            scores['chicken'] += 2.0
            scores['beef'] += 2.0
            scores['pork'] += 2.0
        if 80 < r < 140 and 60 < g < 120 and 50 < b < 110:
            scores['chicken'] += 2.0
            scores['beef'] += 1.5
            scores['pork'] += 1.5
        if r > g and r > b and avg_brightness < 110:
            scores['beef'] += 1.5
            scores['pork'] += 1.5
        
        # Рыба: белая/розовая, средняя яркость
        if avg_brightness > 120 and color_variance < 25:
            scores['fish'] += 2.0
        if r > 140 and g > 130 and b > 120:
            scores['fish'] += 2.0
        
        # Паста: желтая/бежевая, средняя яркость
        if 120 < avg_brightness < 160 and r > 150 and g > 140:
            scores['pasta'] += 2.0
        if r_dominance > 0.35 and g_dominance > 0.35:
            scores['pasta'] += 1.5
        
        # Поиск лучшего совпадения
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Fallback если нет сильного совпадения
        if best_score < 2.0:
            if avg_brightness > 150 and area_pixels < 50000:
                best_type = 'rice'
            elif g_dominance > 0.3 and avg_brightness > 100:
                best_type = 'cabbage'
            elif r > 150 and g > 100:
                best_type = 'carrot' if aspect_ratio < 0.7 else 'tomato'
            elif 80 < avg_brightness < 120:
                best_type = 'chicken'
            else:
                best_type = 'default'
        
        height_cm = FOOD_HEIGHT_MAP.get(best_type, FOOD_HEIGHT_MAP['default'])
        
        logger.debug(
            f"Тип продукта определен: {best_type} (оценка: {best_score:.2f}, "
            f"яркость: {avg_brightness:.1f}, площадь: {area_pixels}, "
            f"цвет: R={r:.1f}, G={g:.1f}, B={b:.1f})"
        )
        
        return best_type, height_cm
    
    def create_mesh_from_polygon(
        self,
        polygon: np.ndarray,
        image_size: Tuple[int, int],
        height_cm: float,
        pixels_per_cm: float
    ) -> o3d.geometry.TriangleMesh:
        """
        Создание 3D меша из 2D полигона
        
        Args:
            polygon: Полигон объекта
            image_size: Размер изображения (ширина, высота)
            height_cm: Высота в см
            pixels_per_cm: Пикселей на см
            
        Returns:
            3D меш
        """
        w, h = image_size
        
        # Упрощение полигона если слишком много точек
        if len(polygon) > 100:
            epsilon = 0.002 * cv2.arcLength(polygon, True)
            polygon = cv2.approxPolyDP(polygon, epsilon, True).squeeze()
            if len(polygon.shape) == 1:
                polygon = polygon.reshape(-1, 2)
        
        vertices_2d = polygon.astype(np.float32)
        vertices_2d[:, 0] = (vertices_2d[:, 0] - w/2) / pixels_per_cm / 100
        vertices_2d[:, 1] = (vertices_2d[:, 1] - h/2) / pixels_per_cm / 100
        
        height_m = height_cm / 100
        center_2d = np.mean(vertices_2d, axis=0)
        
        bottom_vertices = np.hstack([vertices_2d, np.zeros((len(vertices_2d), 1))])
        
        top_vertices = []
        max_dist = max(np.linalg.norm(v - center_2d) for v in vertices_2d)
        
        for v in vertices_2d:
            dist_from_center = np.linalg.norm(v - center_2d)
            height_factor = np.exp(-(dist_from_center / max_dist) ** 2)
            z = height_m * height_factor
            top_vertices.append([v[0], v[1], z])
        
        top_vertices = np.array(top_vertices)
        center_top = np.array([[center_2d[0], center_2d[1], height_m]])
        
        vertices = np.vstack([bottom_vertices, top_vertices, center_top])
        
        n = len(vertices_2d)
        center_idx = 2 * n
        
        triangles = []
        
        # Боковые грани
        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([i, next_i, i + n])
            triangles.append([next_i, next_i + n, i + n])
        
        # Нижняя грань
        for i in range(1, n - 1):
            triangles.append([0, i, i + 1])
        
        # Верхняя грань
        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([center_idx, i + n, next_i + n])
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.6, 0.4, 0.8])
        
        return mesh
    
    def calculate_volume(self, mesh: o3d.geometry.TriangleMesh) -> float:
        """
        Расчет объема меша в см³
        
        Args:
            mesh: 3D меш
            
        Returns:
            Объем в см³
            
        Raises:
            VolumeCalculationError: Если не удалось рассчитать объем
        """
        try:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            volume = 0.0
            for tri in triangles:
                v0, v1, v2 = vertices[tri]
                volume += np.dot(v0, np.cross(v1, v2)) / 6.0
            
            return abs(volume) * 1_000_000
        
        except Exception as e:
            raise VolumeCalculationError(f"Ошибка расчета объема: {e}")
    
    def process_image(
        self,
        image_path: str,
        label_path: str,
        visualize: bool = False
    ) -> List[FoodResult]:
        """
        Обработка одного изображения с файлом labels
        
        Args:
            image_path: Путь к изображению
            label_path: Путь к файлу labels
            visualize: Показывать ли визуализацию
            
        Returns:
            Список результатов обработки
            
        Raises:
            ImageProcessingError: Если не удалось обработать изображение
        """
        try:
            # Загрузка изображения
            image = cv2.imread(image_path)
            if image is None:
                raise ImageProcessingError(f"Не удалось загрузить изображение: {image_path}")
            
            h, w = image.shape[:2]
            
            # Парсинг YOLO labels
            objects = parse_yolo_segmentation(label_path, w, h)
            
            # Поиск тарелки для калибровки
            plate_polygon = None
            pixels_per_cm = None
            
            for obj in objects:
                if obj['class_name'] == 'plate' and len(obj['polygon']) >= 3:
                    plate_polygon = obj['polygon']
                    pixels_per_cm = self.calibrate_from_plate(
                        plate_polygon, w, h, self.plate_diameter_cm
                    )
                    logger.info(f"Тарелка обнаружена: {pixels_per_cm:.2f} пикселей/см")
                    break
            
            # Fallback если тарелка не найдена
            if pixels_per_cm is None:
                plate_diameter_pixels = w * 0.7
                pixels_per_cm = plate_diameter_pixels / self.plate_diameter_cm
                logger.warning(
                    f"Тарелка не обнаружена, используется масштаб по умолчанию: "
                    f"{pixels_per_cm:.2f} пикселей/см"
                )
            
            results = []
            meshes = []
            result_polygons = []
            
            # Обработка каждого объекта 'food'
            for idx, obj in enumerate(objects):
                if obj['class_name'] != 'food':
                    continue
                
                polygon = obj['polygon']
                if len(polygon) < 3:
                    continue
                
                # Создание маски
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 255)
                
                # Определение типа продукта
                food_type, height_cm = self.detect_food_type(polygon, image, mask, w, h)
                
                try:
                    # Создание 3D меша
                    mesh = self.create_mesh_from_polygon(
                        polygon, (w, h), height_cm, pixels_per_cm
                    )
                    
                    # Расчет объема и веса
                    volume_cm3 = self.calculate_volume(mesh)
                    density = FOOD_DENSITY.get(food_type.lower(), FOOD_DENSITY['default'])
                    weight_g = volume_cm3 * density
                    
                    # Центр объекта
                    center_x = np.mean(polygon[:, 0]) / w
                    center_y = np.mean(polygon[:, 1]) / h
                    
                    # Создание результата
                    result = FoodResult(
                        image_path=str(image_path),
                        object_id=idx,
                        class_id=obj['class_id'],
                        food_type=food_type,
                        volume_cm3=round(volume_cm3, 2),
                        weight_g=round(weight_g, 2),
                        weight_kg=round(weight_g / 1000, 4),
                        density_g_cm3=density,
                        polygon_points=len(polygon),
                        area_pixels=int(np.sum(mask > 0)),
                        center_x=round(center_x, 3),
                        center_y=round(center_y, 3)
                    )
                    
                    results.append(result)
                    result_polygons.append(polygon)
                    
                    if visualize:
                        meshes.append((mesh, result))
                
                except Exception as e:
                    logger.error(f"Ошибка обработки объекта {idx}: {e}")
            
            if visualize and meshes:
                logger.info(f"Показ визуализации изображения с {len(meshes)} объектами...")
                self._visualize_image_with_segments(image, results, result_polygons)
                logger.info(f"Показ 3D моделей для {len(meshes)} объектов...")
                self._visualize_meshes(meshes)
            
            return results
        
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(f"Ошибка обработки изображения: {e}")
    
    def _visualize_image_with_segments(
        self,
        image: np.ndarray,
        results: List[FoodResult],
        polygons: List[np.ndarray],
        output_path: Optional[str] = None
    ):
        """Визуализация изображения с сегментацией"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MPLPolygon
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        colors = [
            [1.0, 0.0, 0.0, 0.4],  # Красный
            [0.0, 1.0, 0.0, 0.4],  # Зеленый
            [0.0, 0.0, 1.0, 0.4],  # Синий
            [1.0, 1.0, 0.0, 0.4],  # Желтый
            [1.0, 0.0, 1.0, 0.4],  # Пурпурный
            [0.0, 1.0, 1.0, 0.4],  # Голубой
        ]
        
        for idx, (result, polygon) in enumerate(zip(results, polygons)):
            if polygon is not None and len(polygon) > 0:
                color = colors[idx % len(colors)]
                poly_points = [(p[0], p[1]) for p in polygon]
                poly = MPLPolygon(
                    poly_points,
                    closed=True,
                    fill=True,
                    facecolor=color,
                    edgecolor=color[:3],
                    linewidth=2
                )
                ax.add_patch(poly)
                
                center_x = int(result.center_x * image.shape[1])
                center_y = int(result.center_y * image.shape[0])
                label = f"{result.food_type}\n{result.weight_g:.1f}g"
                ax.text(
                    center_x, center_y, label,
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    ha='center', va='center'
                )
        
        total_weight = sum(r.weight_g for r in results)
        ax.set_title(
            f"Обнаружено объектов: {len(results)}, Общий вес: {total_weight:.1f}g",
            fontsize=14, fontweight='bold'
        )
        ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Визуализация сохранена: {output_path}")
        
        plt.show()
        plt.close()
    
    def _visualize_meshes(self, meshes: List[Tuple]):
        """Визуализация 3D мешей"""
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=[0, 0, 0]
        )
        
        colors = [
            [0.6, 0.4, 0.8],  # Фиолетовый
            [0.8, 0.4, 0.6],  # Розовый
            [0.4, 0.8, 0.6],  # Зеленый
            [0.8, 0.6, 0.4],  # Оранжевый
        ]
        
        geometries = [coord_frame]
        
        for i, (mesh, result) in enumerate(meshes):
            mesh_copy = o3d.geometry.TriangleMesh(mesh)
            mesh_copy.paint_uniform_color(colors[i % len(colors)])
            offset = np.array([i * 0.15, 0, 0])
            mesh_copy.translate(offset)
            geometries.append(mesh_copy)
        
        weights = [f"{r.food_type}:{r.weight_g:.0f}g" for _, r in meshes]
        window_name = " | ".join(weights)
        
        logger.info("3D окно открыто. Закройте окно для продолжения...")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1024,
            height=768,
            mesh_show_back_face=True
        )

