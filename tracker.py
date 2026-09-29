"""
Object Tracker Module using Ultralytics YOLO Tracking API.
Supports ByteTrack and BoT-SORT real-time multi-object tracking algorithm configurations,
maintains unique object ID persistence across video frames, and tallies tracking statistics.
"""

import os
import cv2
import numpy as np
from detector import YOLODetector
from utils import get_device


class YOLOTracker:
    """
    Real-time Multi-Object Tracker wrapping Ultralytics YOLO `.track()` engine.
    Supports ByteTrack & BoT-SORT algorithm configurations with ID persistence.
    """

    def __init__(self, detector=None, model_name="yolov8n.pt", tracker_type="bytetrack"):
        """
        Initialize YOLOTracker.
        
        Args:
            detector (YOLODetector or str or None): Existing detector instance, model_name string, or None.
            model_name (str): YOLO model weight file.
            tracker_type (str): 'bytetrack' or 'botsort'.
        """
        if isinstance(detector, str):
            model_name = detector
            detector = None

        if detector is not None:
            self.detector = detector
        else:
            self.detector = YOLODetector(model_name=model_name)
            
        self.tracker_type = tracker_type.lower()
        self.tracker_config = f"{self.tracker_type}.yaml"
        
        # Unique Track ID persistence set across video stream session
        self.unique_track_ids = set()
        self.active_objects_count = 0
        self.class_counts = {}

    def set_tracker_type(self, tracker_type):
        """Update tracker algorithm (bytetrack or botsort)."""
        valid_trackers = ["bytetrack", "botsort"]
        t_type = tracker_type.lower()
        if t_type in valid_trackers:
            self.tracker_type = t_type
            self.tracker_config = f"{self.tracker_type}.yaml"

    def reset(self):
        """Reset stream tracking history state."""
        self.unique_track_ids.clear()
        self.active_objects_count = 0
        self.class_counts.clear()

    def track(self, frame, conf_thresh=0.25, iou_thresh=0.45, tracker_type=None, target_classes=None):
        """
        Track objects across frame sequence using persistent ID assignment.
        
        Args:
            frame (np.ndarray): Input BGR image frame.
            conf_thresh (float): Detection confidence threshold.
            iou_thresh (float): IoU NMS threshold.
            tracker_type (str or None): Override tracker type ('bytetrack' / 'botsort').
            target_classes (list or None): Filter classes to track.
            
        Returns:
            list of dict: Tracked object detections with persistent 'track_id':
                {
                    'bbox': [x1, y1, x2, y2],
                    'class_id': int,
                    'class_name': str,
                    'confidence': float,
                    'track_id': int or None
                }
        """
        if frame is None or frame.size == 0:
            return []

        if tracker_type is not None:
            self.set_tracker_type(tracker_type)

        try:
            # Execute Ultralytics tracking stream inference with persistence flag
            results = self.detector.model.track(
                source=frame,
                conf=conf_thresh,
                iou=iou_thresh,
                tracker=self.tracker_config,
                persist=True,
                device=self.detector.device,
                verbose=False
            )

            tracked_objects = []
            self.class_counts.clear()

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                xyxy_tensor = boxes.xyxy.cpu().numpy()
                conf_tensor = boxes.conf.cpu().numpy()
                cls_tensor = boxes.cls.cpu().numpy()
                
                # Extract track IDs if assigned by ByteTrack/BoT-SORT
                track_ids = None
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy()

                for i in range(len(boxes)):
                    bbox = xyxy_tensor[i].tolist()
                    confidence = float(conf_tensor[i])
                    class_id = int(cls_tensor[i])
                    class_name = self.detector.class_names.get(class_id, f"cls_{class_id}")
                    
                    track_id = int(track_ids[i]) if (track_ids is not None and i < len(track_ids)) else None

                    # Class filter
                    if target_classes is not None and len(target_classes) > 0:
                        if class_name not in target_classes and class_id not in target_classes:
                            continue

                    if track_id is not None:
                        self.unique_track_ids.add(track_id)

                    self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

                    tracked_objects.append({
                        'bbox': bbox,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'track_id': track_id
                    })

            self.active_objects_count = len(tracked_objects)
            return tracked_objects

        except Exception as e:
            print(f"[YOLOTracker] Tracking error: {str(e)}")
            # Fallback to detector if tracker throws exception
            return self.detector.detect(
                frame=frame,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                target_classes=target_classes
            )

    def get_stats(self):
        """
        Return tracking statistics dictionary.
        """
        return {
            'active_objects': self.active_objects_count,
            'total_unique_tracked': len(self.unique_track_ids),
            'class_distribution': dict(self.class_counts)
        }
