"""
Object Detector Module using Ultralytics YOLOv8.
Handles model loading, device placement, inference execution, and detection parsing.
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utils import get_device


class YOLODetector:
    """
    YOLO Object Detector wrapping Ultralytics API.
    Provides robust detection for images and video frames.
    """

    def __init__(self, model_name="yolov8n.pt", models_dir="models"):
        """
        Initialize YOLO model detector.
        
        Args:
            model_name (str): Model file name or path (e.g., 'yolov8n.pt').
            models_dir (str): Folder path to store/load YOLO models.
        """
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_name = model_name
        
        # Resolve path to models directory if relative name given
        if not os.path.isabs(model_name) and not os.path.exists(model_name):
            model_path = os.path.join(self.models_dir, model_name)
        else:
            model_path = model_name

        self.device, self.device_display = get_device()
        self.model = None
        self.class_names = {}
        self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load YOLO weights onto target hardware device.
        """
        try:
            # Instantiate YOLO model (Ultralytics handles download if not present)
            self.model = YOLO(model_path)
            
            # Extract class names mapping (dict of id -> name)
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
            else:
                self.class_names = {}

            print(f"[YOLODetector] Loaded model '{self.model_name}' on device '{self.device_display}'.")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{model_path}': {str(e)}")

    def detect(self, frame, conf_thresh=0.25, iou_thresh=0.45, target_classes=None):
        """
        Perform object detection on a single frame.
        
        Args:
            frame (np.ndarray): Input image/frame in BGR format.
            conf_thresh (float): Minimum confidence threshold (0.0 to 1.0).
            iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS).
            target_classes (list or None): Filter detection by class IDs or names.
            
        Returns:
            list of dict: List of detections where each item is:
                {
                    'bbox': [x1, y1, x2, y2],
                    'class_id': int,
                    'class_name': str,
                    'confidence': float
                }
        """
        if frame is None or frame.size == 0:
            return []

        try:
            # Run inference
            results = self.model.predict(
                source=frame,
                conf=conf_thresh,
                iou=iou_thresh,
                device=self.device,
                verbose=False
            )

            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Extract numpy arrays
                xyxy_tensor = boxes.xyxy.cpu().numpy()
                conf_tensor = boxes.conf.cpu().numpy()
                cls_tensor = boxes.cls.cpu().numpy()

                for i in range(len(boxes)):
                    bbox = xyxy_tensor[i].tolist()
                    confidence = float(conf_tensor[i])
                    class_id = int(cls_tensor[i])
                    class_name = self.class_names.get(class_id, f"cls_{class_id}")

                    # Filter by target_classes if specified
                    if target_classes is not None and len(target_classes) > 0:
                        if class_name not in target_classes and class_id not in target_classes:
                            continue

                    detections.append({
                        'bbox': bbox,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence
                    })

            return detections

        except Exception as e:
            print(f"[YOLODetector] Detection error: {str(e)}")
            return []

    def get_class_names(self):
        """Get dictionary or list of class names."""
        return self.class_names
