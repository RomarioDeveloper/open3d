"""
Food Weight Estimation Service
Processes images with YOLO segmentation labels and calculates 3D volume/weight
"""

import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Food density database (g/cm³)
FOOD_DENSITY = {
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

CLASS_NAMES = {0: 'plate', 1: 'food', 2: 'glass'}


@dataclass
class FoodResult:
    """Result data structure"""
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


class FoodWeightEstimator:
    """Main estimator class"""
    
    def __init__(self, plate_diameter_cm: float = 24.0):
        self.plate_diameter_cm = plate_diameter_cm
        
    def parse_yolo_segmentation(self, label_path: str, img_width: int, img_height: int) -> List[Dict]:
        """Parse YOLO segmentation format"""
        objects = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                
                class_id = int(parts[0])
                coords = list(map(float, parts[1:-1]))  # Skip confidence score
                
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
        
        return objects
    
    def calibrate_from_plate(self, plate_polygon: np.ndarray, img_width: int, img_height: int,
                            known_plate_diameter_cm: float) -> float:
        """
        Calibrate scale from plate detection
        Returns pixels_per_cm based on detected plate size
        """
        # Find bounding box of plate
        bbox = cv2.boundingRect(plate_polygon)
        plate_width_pixels = bbox[2]
        plate_height_pixels = bbox[3]
        
        # Use average of width and height as diameter estimate
        plate_diameter_pixels = (plate_width_pixels + plate_height_pixels) / 2.0
        
        # Calculate pixels per cm
        pixels_per_cm = plate_diameter_pixels / known_plate_diameter_cm
        
        return pixels_per_cm
    
    def detect_food_type(self, polygon: np.ndarray, image: np.ndarray, mask: np.ndarray, 
                        img_width: int, img_height: int) -> Tuple[str, float]:
        """
        Detect food type from segmentation
        Uses position, color, and shape characteristics
        """
        h, w = image.shape[:2]
        center_x = np.mean(polygon[:, 0]) / w
        center_y = np.mean(polygon[:, 1]) / h
        
        # Extract color from masked region
        masked_region = image[mask > 0]
        if len(masked_region) > 0:
            avg_color = np.mean(masked_region, axis=0)
            avg_brightness = np.mean(avg_color)
        else:
            avg_brightness = 128
        
        # Calculate area and aspect ratio
        area_pixels = np.sum(mask > 0)
        bbox = cv2.boundingRect(polygon)
        aspect_ratio = bbox[2] / max(bbox[3], 1)
        
        # Decision logic
        # Left side + bright = rice
        if center_x < 0.5 and avg_brightness > 150:
            return 'rice', 2.0
        
        # Right side + darker = meat/cabbage
        elif center_x > 0.5 and avg_brightness < 120:
            return 'cabbage', 4.0
        
        # Large area + round = cabbage
        elif area_pixels > 50000 and aspect_ratio > 0.8:
            return 'cabbage', 4.0
        
        # Small area + bright = rice
        elif area_pixels < 50000 and avg_brightness > 150:
            return 'rice', 2.0
        
        # Default
        return 'cabbage', 4.0
    
    def create_mesh_from_polygon(self, polygon: np.ndarray, image_size: Tuple[int, int],
                                height_cm: float, pixels_per_cm: float) -> o3d.geometry.TriangleMesh:
        """Create 3D mesh from 2D polygon"""
        w, h = image_size
        
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
        
        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([i, next_i, i + n])
            triangles.append([next_i, next_i + n, i + n])
        
        for i in range(1, n - 1):
            triangles.append([0, i, i + 1])
        
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
        """Calculate mesh volume in cm³"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        volume = 0.0
        for tri in triangles:
            v0, v1, v2 = vertices[tri]
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        
        return abs(volume) * 1_000_000
    
    def process_image(self, image_path: str, label_path: str, visualize: bool = False) -> List[FoodResult]:
        """Process single image with label file"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        h, w = image.shape[:2]
        objects = self.parse_yolo_segmentation(label_path, w, h)
        
        # Find plate for calibration
        plate_polygon = None
        pixels_per_cm = None
        
        for obj in objects:
            if obj['class_name'] == 'plate' and len(obj['polygon']) >= 3:
                plate_polygon = obj['polygon']
                pixels_per_cm = self.calibrate_from_plate(plate_polygon, w, h, self.plate_diameter_cm)
                logger.info(f"Plate detected: {pixels_per_cm:.2f} pixels/cm")
                break
        
        # Fallback to default if no plate found
        if pixels_per_cm is None:
            plate_diameter_pixels = w * 0.7
            pixels_per_cm = plate_diameter_pixels / self.plate_diameter_cm
            logger.warning(f"No plate detected, using default scale: {pixels_per_cm:.2f} pixels/cm")
        
        results = []
        meshes = []
        
        for idx, obj in enumerate(objects):
            if obj['class_name'] != 'food':
                continue
            
            polygon = obj['polygon']
            if len(polygon) < 3:
                continue
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            
            food_type, height_cm = self.detect_food_type(polygon, image, mask, w, h)
            
            try:
                mesh = self.create_mesh_from_polygon(polygon, (w, h), height_cm, pixels_per_cm)
                volume_cm3 = self.calculate_volume(mesh)
                
                density = FOOD_DENSITY.get(food_type.lower(), FOOD_DENSITY['default'])
                weight_g = volume_cm3 * density
                
                center_x = np.mean(polygon[:, 0]) / w
                center_y = np.mean(polygon[:, 1]) / h
                
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
                if visualize:
                    meshes.append((mesh, result))
                
            except Exception as e:
                logger.error(f"Error processing object {idx}: {e}")
        
        if visualize and meshes:
            logger.info(f"Opening 3D visualization for {len(meshes)} objects...")
            self._visualize_meshes(meshes)
        
        return results
    
    def _visualize_meshes(self, meshes: List[Tuple]):
        """Visualize all 3D meshes together"""
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=[0, 0, 0]
        )
        
        colors = [
            [0.6, 0.4, 0.8],  # Purple
            [0.8, 0.4, 0.6],  # Pink
            [0.4, 0.8, 0.6],  # Green
            [0.8, 0.6, 0.4],  # Orange
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
        
        logger.info(f"3D window opened. Close window to continue...")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1024,
            height=768,
            mesh_show_back_face=True
        )


def process_directory(input_dir: str = './images', labels_dir: str = './labels', 
                     output_dir: str = './results', plate_diameter_cm: float = 24.0,
                     visualize: bool = False):
    """
    Process all images in directory
    
    Args:
        input_dir: Directory with images
        labels_dir: Directory with YOLO label files
        output_dir: Directory for JSON results
        plate_diameter_cm: Reference plate diameter
        visualize: Show 3D visualization for each image
    """
    input_path = Path(input_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    if not labels_path.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images")
    
    estimator = FoodWeightEstimator(plate_diameter_cm=plate_diameter_cm)
    all_results = {}
    
    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            logger.warning(f"No label file for {img_file.name}")
            continue
        
        try:
            results = estimator.process_image(str(img_file), str(label_file), visualize=visualize)
            
            if results:
                all_results[img_file.name] = [asdict(r) for r in results]
                
                # Save individual JSON
                output_file = output_path / f"{img_file.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'image': str(img_file.name),
                        'objects': [asdict(r) for r in results],
                        'total_weight_g': sum(r.weight_g for r in results),
                        'total_objects': len(results)
                    }, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Processed {img_file.name}: {len(results)} objects, "
                          f"total weight: {sum(r.weight_g for r in results):.1f}g")
            
        except Exception as e:
            logger.error(f"Failed {img_file.name}: {e}")
    
    # Save summary
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
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Summary: {len(all_results)} images, "
              f"{sum(len(objs) for objs in all_results.values())} objects")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Food Weight Estimation Service')
    parser.add_argument('--input', '-i', default='./images', help='Input images directory')
    parser.add_argument('--labels', '-l', default='./labels', help='Labels directory')
    parser.add_argument('--output', '-o', default='./results', help='Output directory')
    parser.add_argument('--plate-diameter', '-d', type=float, default=24.0, 
                       help='Plate diameter in cm')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Show 3D visualization for each image')
    
    args = parser.parse_args()
    
    process_directory(
        input_dir=args.input,
        labels_dir=args.labels,
        output_dir=args.output,
        plate_diameter_cm=args.plate_diameter,
        visualize=args.visualize
    )
