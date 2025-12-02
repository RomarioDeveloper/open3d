"""
Food Weight Estimation from YOLO Segmentation Labels
Creates accurate 3D mesh from segmentation polygons and calculates weight
"""

import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Food density database (g/cm³)
FOOD_DENSITY = {
    'food': 0.70,      # Class 0 - general food
    'plate': 0.00,     # Class 1 - plate (ignore)
    'glass': 0.00,     # Class 2 - glass (ignore)
    'cabbage': 0.55,
    'potato': 0.75,
    'carrot': 0.64,
    'tomato': 0.60,
    'cucumber': 0.65,
    'chicken': 1.05,
    'beef': 1.06,
    'pork': 1.04,
    'fish': 1.05,
    'rice': 0.85,
    'pasta': 0.70,
    'default': 0.70,
}

# Class names
CLASS_NAMES = {
    0: 'plate',
    1: 'food',
    2: 'glass',
}


def parse_yolo_segmentation(label_path: str, img_width: int, img_height: int) -> List[Dict]:
    """
    Parse YOLO segmentation format
    Returns list of objects with class and polygon points
    """
    objects = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least class + 3 points (6 coords)
                continue
            
            class_id = int(parts[0])
            # Last value is confidence score, skip it
            coords = list(map(float, parts[1:-1]))
            
            # Convert normalized coords to pixel coords
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = int(coords[i] * img_width)
                    y = int(coords[i + 1] * img_height)
                    points.append([x, y])
            
            objects.append({
                'class_id': class_id,
                'class_name': CLASS_NAMES.get(class_id, f'class_{class_id}'),
                'polygon': np.array(points, dtype=np.int32)
            })
    
    return objects


class FoodWeight3D:
    """Main class for 3D reconstruction and weight estimation from YOLO labels"""
    
    def __init__(self, plate_diameter_cm: float = 24.0):
        """
        Initialize analyzer
        
        Args:
            plate_diameter_cm: Reference plate diameter in cm for scale
        """
        self.plate_diameter_cm = plate_diameter_cm
        self.mesh = None
        self.pointcloud = None
        
    def process_image_with_labels(
        self,
        image_path: str,
        label_path: str,
        food_type: str = 'default',
        estimated_height_cm: float = 4.0,
        visualize: bool = True
    ) -> List[Dict]:
        """
        Process image using YOLO segmentation labels
        
        Args:
            image_path: Path to input image
            label_path: Path to YOLO label file
            food_type: Type of food for density lookup
            estimated_height_cm: Estimated height of food in cm
            visualize: Show 3D visualization
            
        Returns:
            List of analysis results for each food object
        """
        logger.info(f"Processing: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        h, w = image.shape[:2]
        logger.info(f"Image size: {w}x{h}")
        
        # Parse labels
        objects = parse_yolo_segmentation(label_path, w, h)
        logger.info(f"Found {len(objects)} objects")
        
        # Process each food object
        results = []
        
        for idx, obj in enumerate(objects):
            class_name = obj['class_name']
            logger.info(f"\nObject {idx+1}: {class_name} (class_id={obj['class_id']})")
            
            # Skip non-food objects
            if class_name in ['plate', 'glass']:
                logger.info(f"  Skipping {class_name}")
                continue
            
            # Determine food type based on position and size
            # Left side (x < 0.5) is usually rice, right side is meat/cabbage
            polygon = obj['polygon']
            center_x = np.mean(polygon[:, 0]) / w
            
            if center_x < 0.5:
                detected_type = 'rice'
                height_cm = 2.0  # Rice is flatter
            else:
                detected_type = 'cabbage'  # or meat
                height_cm = 4.0
            
            logger.info(f"  Detected type: {detected_type} (center_x={center_x:.2f})")
            
            if len(polygon) < 3:
                logger.warning(f"  Insufficient points: {len(polygon)}")
                continue
            
            # Create mask from polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            
            area_pixels = np.sum(mask > 0)
            logger.info(f"  Polygon points: {len(polygon)}")
            logger.info(f"  Area: {area_pixels} pixels")
            
            try:
                # Create 3D mesh with appropriate height
                mesh = self._create_mesh_from_polygon(
                    polygon, 
                    (w, h),
                    height_cm,  # Use detected height
                    self.plate_diameter_cm
                )
                
                logger.info(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                
                # Calculate volume
                volume_cm3 = self._calculate_volume(mesh)
                logger.info(f"  Volume: {volume_cm3:.2f} cm³")
                
                # Use detected type if food_type is default
                actual_food_type = detected_type if food_type == 'default' else food_type
                
                # Calculate weight
                density = FOOD_DENSITY.get(actual_food_type.lower(), FOOD_DENSITY['default'])
                weight_g = volume_cm3 * density
                
                result = {
                    'object_id': idx,
                    'class_name': class_name,
                    'food_type': actual_food_type,
                    'detected_type': detected_type,
                    'volume_cm3': round(volume_cm3, 2),
                    'weight_g': round(weight_g, 2),
                    'weight_kg': round(weight_g / 1000, 4),
                    'density_g_cm3': density,
                    'polygon_points': len(polygon),
                    'area_pixels': int(area_pixels),
                }
                
                logger.info(f"  Weight: {weight_g:.2f} g ({weight_g/1000:.3f} kg)")
                
                results.append(result)
                
                # Visualize
                if visualize:
                    self._visualize(mesh, result, image, mask)
                
            except Exception as e:
                logger.error(f"  Error processing: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def _create_mesh_from_polygon(
        self, 
        polygon: np.ndarray,
        image_size: Tuple[int, int],
        height_cm: float,
        plate_diameter_cm: float
    ) -> o3d.geometry.TriangleMesh:
        """
        Create 3D mesh from 2D polygon using extrusion with realistic dome shape
        """
        w, h = image_size
        
        # Simplify polygon if too many points
        if len(polygon) > 100:
            epsilon = 0.002 * cv2.arcLength(polygon, True)
            polygon = cv2.approxPolyDP(polygon, epsilon, True).squeeze()
            if len(polygon.shape) == 1:
                polygon = polygon.reshape(-1, 2)
        
        # Scale to real-world coordinates
        plate_diameter_pixels = w * 0.7
        pixels_per_cm = plate_diameter_pixels / plate_diameter_cm
        
        # Convert to meters and center
        vertices_2d = polygon.astype(np.float32)
        vertices_2d[:, 0] = (vertices_2d[:, 0] - w/2) / pixels_per_cm / 100
        vertices_2d[:, 1] = (vertices_2d[:, 1] - h/2) / pixels_per_cm / 100
        
        # Create 3D vertices with realistic dome/mound shape
        height_m = height_cm / 100
        center_2d = np.mean(vertices_2d, axis=0)
        
        # Bottom vertices (z=0)
        bottom_vertices = np.hstack([vertices_2d, np.zeros((len(vertices_2d), 1))])
        
        # Top vertices with smooth dome shape
        top_vertices = []
        max_dist = max(np.linalg.norm(v - center_2d) for v in vertices_2d)
        
        for v in vertices_2d:
            dist_from_center = np.linalg.norm(v - center_2d)
            
            # Smooth gaussian-like dome shape
            height_factor = np.exp(-(dist_from_center / max_dist) ** 2)
            z = height_m * height_factor
            
            top_vertices.append([v[0], v[1], z])
        
        top_vertices = np.array(top_vertices)
        
        # Add center point at peak for better shape
        center_top = np.array([[center_2d[0], center_2d[1], height_m]])
        
        # Combine all vertices
        vertices = np.vstack([bottom_vertices, top_vertices, center_top])
        
        n = len(vertices_2d)
        center_idx = 2 * n
        
        # Create triangles
        triangles = []
        
        # Side faces
        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([i, next_i, i + n])
            triangles.append([next_i, next_i + n, i + n])
        
        # Bottom cap (fan from first vertex)
        for i in range(1, n - 1):
            triangles.append([0, i, i + 1])
        
        # Top cap (fan from center peak)
        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([center_idx, i + n, next_i + n])
        
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.6, 0.4, 0.8])  # Purple like in image
        
        return mesh
    
    def _calculate_volume(self, mesh: o3d.geometry.TriangleMesh) -> float:
        """Calculate mesh volume in cm³ using divergence theorem"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        volume = 0.0
        for tri in triangles:
            v0, v1, v2 = vertices[tri]
            # Signed volume of tetrahedron
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        
        # Convert m³ to cm³
        return abs(volume) * 1_000_000
    
    def _visualize(
        self, 
        mesh: o3d.geometry.TriangleMesh, 
        result: Dict,
        image: np.ndarray,
        mask: np.ndarray
    ):
        """Visualize 3D mesh and save overlay image"""
        
        # Save overlay image
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [200, 100, 255]  # Purple
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
        
        # Add text
        text = f"{result['food_type']}: {result['weight_g']:.1f}g"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 255), 2)
        
        output_path = f"output_{result['object_id']}.jpg"
        cv2.imwrite(output_path, overlay)
        logger.info(f"  Saved overlay: {output_path}")
        
        # 3D visualization
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=[0, 0, 0]
        )
        
        o3d.visualization.draw_geometries(
            [mesh, coord_frame],
            window_name=f"{result['food_type']} - {result['weight_g']:.0f}g",
            width=1024,
            height=768,
            mesh_show_back_face=True
        )


def process_directory(
    input_dir: str = '.',
    labels_dir: str = './labels',
    output_file: str = 'results.txt',
    food_type: str = 'cabbage',
    height_cm: float = 4.0,
    visualize: bool = True
):
    """
    Process all images with corresponding label files
    
    Args:
        input_dir: Directory containing images
        labels_dir: Directory containing YOLO label files
        output_file: Output file for results
        food_type: Type of food
        height_cm: Estimated height in cm
        visualize: Show 3D visualization for each
    """
    input_path = Path(input_dir)
    labels_path = Path(labels_dir)
    
    if not labels_path.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    image_files = [
        f for f in input_path.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images")
    
    analyzer = FoodWeight3D(plate_diameter_cm=24.0)
    all_results = []
    
    for img_file in image_files:
        # Find corresponding label file
        label_file = labels_path / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            logger.warning(f"No label file for {img_file.name}")
            continue
        
        try:
            results = analyzer.process_image_with_labels(
                str(img_file),
                str(label_file),
                food_type=food_type,
                estimated_height_cm=height_cm,
                visualize=visualize
            )
            
            all_results.extend(results)
            logger.info(f"Success: {img_file.name} - {len(results)} objects")
            
        except Exception as e:
            logger.error(f"Failed {img_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if all_results:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Food Weight Analysis Results (YOLO Segmentation)\n")
            f.write("=" * 80 + "\n\n")
            
            for r in all_results:
                f.write(f"Class: {r['class_name']}\n")
                f.write(f"Food Type: {r['food_type']}\n")
                f.write(f"Volume: {r['volume_cm3']} cm³\n")
                f.write(f"Weight: {r['weight_g']} g ({r['weight_kg']} kg)\n")
                f.write(f"Density: {r['density_g_cm3']} g/cm³\n")
                f.write(f"Polygon Points: {r['polygon_points']}\n")
                f.write(f"Area: {r['area_pixels']} pixels\n")
                f.write("-" * 80 + "\n")
        
        logger.info(f"\nResults saved to {output_file}")
        
        # Summary
        total_weight = sum(r['weight_g'] for r in all_results)
        avg_weight = total_weight / len(all_results) if all_results else 0
        
        logger.info(f"\nSummary:")
        logger.info(f"  Processed: {len(all_results)} objects")
        logger.info(f"  Total weight: {total_weight:.1f} g")
        logger.info(f"  Average weight: {avg_weight:.1f} g")


if __name__ == '__main__':
    # Process all images with YOLO labels
    process_directory(
        input_dir='.',
        labels_dir='./labels',
        output_file='results.txt',
        food_type='cabbage',
        height_cm=4.0,
        visualize=True
    )
