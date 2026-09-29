"""
Utility functions for Vision-Based Object Detection & Tracking Using YOLO.
Provides hardware detection, FPS calculation, crisp visual annotations,
video processing utilities, and output export functions.
"""

import os
import time
import cv2
import numpy as np
import torch
from datetime import datetime

# Distinct color palette (RGB) for aesthetic visual tracking
COLOR_PALETTE = [
    (255, 99, 132),   # Coral Red
    (54, 162, 235),   # Bright Blue
    (255, 206, 86),   # Canary Yellow
    (75, 192, 192),   # Teal
    (153, 102, 255),  # Lavender Purple
    (255, 159, 64),   # Orange
    (46, 204, 113),   # Emerald Green
    (231, 76, 60),    # Alizarin Red
    (155, 89, 182),   # Amethyst Purple
    (52, 152, 219),   # Peter River Blue
    (241, 196, 15),   # Sunflower Yellow
    (26, 188, 156),   # Turquoise
    (230, 126, 34),   # Carrot Orange
    (243, 156, 18),   # Orange Gold
    (142, 68, 173),   # Wisteria
]

def get_device():
    """
    Automatically detect the best available processing hardware device.
    
    Returns:
        tuple: (device_string, device_name)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return "cuda", f"GPU ({device_name})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Silicon GPU (MPS)"
    else:
        return "cpu", "CPU (Standard Processor)"


class FPSCalculator:
    """High-precision FPS counter using exponential moving average for smooth UI rendering."""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.prev_time = None
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
    def update(self):
        """Update FPS count for current frame."""
        curr_time = time.time()
        self.frame_count += 1
        
        if self.prev_time is not None:
            delta = curr_time - self.prev_time
            if delta > 0:
                inst_fps = 1.0 / delta
                if self.fps == 0.0:
                    self.fps = inst_fps
                else:
                    self.fps = (self.alpha * inst_fps) + ((1.0 - self.alpha) * self.fps)
        self.prev_time = curr_time
        return self.fps

    def get_fps(self):
        return round(self.fps, 1)
        
    def get_average_fps(self):
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return round(self.frame_count / elapsed, 1)
        return 0.0


def get_color_for_id(identifier):
    """Generate a consistent, distinctive color based on class ID or track ID."""
    idx = int(identifier) % len(COLOR_PALETTE)
    return COLOR_PALETTE[idx]


def draw_annotations(
    frame,
    objects,
    show_boxes=True,
    show_labels=True,
    show_conf=True,
    show_id=True,
    show_stats_overlay=True,
    fps=0.0
):
    """
    Draw clean bounding boxes, labels, tracking IDs, and performance stats on frame.
    
    Args:
        frame (np.ndarray): Input BGR frame image.
        objects (list): List of dicts with keys 'bbox', 'class_name', 'confidence', optional 'track_id'.
        show_boxes (bool): Whether to draw bounding box rectangles.
        show_labels (bool): Whether to show object class labels.
        show_conf (bool): Whether to show confidence score percentage.
        show_id (bool): Whether to show tracking ID.
        show_stats_overlay (bool): Whether to render top-left stats badge.
        fps (float): Current FPS rate.
        
    Returns:
        np.ndarray: Annotated frame image.
    """
    annotated = frame.copy()
    h, w, _ = annotated.shape

    for obj in objects:
        bbox = obj.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = map(int, bbox)

        # Clip coordinates to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        class_name = obj.get('class_name', 'object')
        conf = obj.get('confidence', 0.0)
        track_id = obj.get('track_id', None)

        # Determine color (by track_id if available, else by class_id)
        color_id = track_id if track_id is not None else obj.get('class_id', 0)
        color = get_color_for_id(color_id)

        # 1. Draw Bounding Box
        if show_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # Optional corner accents for modern aesthetic UI
            corner_length = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_length > 0:
                cv2.line(annotated, (x1, y1), (x1 + corner_length, y1), color, 4)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_length), color, 4)
                cv2.line(annotated, (x2, y1), (x2 - corner_length, y1), color, 4)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_length), color, 4)
                cv2.line(annotated, (x1, y2), (x1 + corner_length, y2), color, 4)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_length), color, 4)
                cv2.line(annotated, (x2, y2), (x2 - corner_length, y2), color, 4)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_length), color, 4)

        # 2. Build Label String
        label_parts = []
        if show_labels:
            label_parts.append(f"{class_name}")
        if show_id and track_id is not None:
            label_parts.append(f"ID: {track_id}")
        if show_conf:
            label_parts.append(f"{int(conf * 100)}%")

        if label_parts:
            label_text = " | ".join(label_parts)
            
            # Label background box settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Position box above bbox, or inside if near top edge
            label_bg_y1 = max(0, y1 - text_h - baseline - 8)
            label_bg_y2 = y1 if y1 - text_h - baseline - 8 >= 0 else y1 + text_h + baseline + 8
            label_bg_x2 = min(w, x1 + text_w + 12)

            # Semi-transparent background badge
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            # Draw label text (white or black depending on color brightness)
            text_y = label_bg_y1 + text_h + 4 if y1 - text_h - baseline - 8 >= 0 else y1 + text_h + 4
            cv2.putText(
                annotated,
                label_text,
                (x1 + 6, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA
            )

    # 3. Draw Performance Overlay (Top-Left HUD)
    if show_stats_overlay:
        hud_text = f"FPS: {fps:.1f} | Objects: {len(objects)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), bl = cv2.getTextSize(hud_text, font, font_scale, thickness)
        
        # Semi-transparent dark overlay box for HUD
        hud_overlay = annotated.copy()
        cv2.rectangle(hud_overlay, (10, 10), (20 + tw, 20 + th + bl), (20, 24, 33), -1)
        cv2.addWeighted(hud_overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (10, 10), (20 + tw, 20 + th + bl), (0, 255, 204), 1)
        
        cv2.putText(
            annotated,
            hud_text,
            (15, 15 + th),
            font,
            font_scale,
            (0, 255, 204),
            thickness,
            lineType=cv2.LINE_AA
        )

    return annotated


class ProcessedVideoWriter:
    """Helper class to record and export processed video streams cleanly."""

    def __init__(self, output_dir="output", fps=25.0, frame_size=(640, 480)):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.file_path = None

    def initialize(self, frame_size, fps=25.0):
        """Initialize VideoWriter with fallback codecs."""
        self.frame_size = frame_size
        self.fps = fps
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.output_dir, f"tracking_output_{timestamp}.mp4")

        # Try mp4v first, then avc1, then XVID
        codecs = ['mp4v', 'avc1', 'XVID', 'MJPG']
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.writer = cv2.VideoWriter(self.file_path, fourcc, self.fps, self.frame_size)
            if self.writer.isOpened():
                break

        if not self.writer or not self.writer.isOpened():
            raise RuntimeError("Failed to initialize OpenCV VideoWriter with available codecs.")
        return self.file_path

    def write_frame(self, frame):
        """Write a BGR image frame."""
        if self.writer and self.writer.isOpened():
            if (frame.shape[1], frame.shape[0]) != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            self.writer.write(frame)

    def release(self):
        """Release video writer file handle."""
        if self.writer:
            self.writer.release()
            self.writer = None
        return self.file_path


def save_screenshot(frame, output_dir="output"):
    """
    Save annotated frame as a PNG screenshot.
    
    Returns:
        str: Absolute or relative filepath of saved image.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"screenshot_{timestamp}.png")
    cv2.imwrite(file_path, frame)
    return file_path


def get_available_models(models_dir="models"):
    """
    List available YOLO model options (downloaded or downloadable).
    """
    os.makedirs(models_dir, exist_ok=True)
    standard_models = [
        "yolov8n.pt",  # Nano - Fastest
        "yolov8s.pt",  # Small - Balanced
        "yolov8m.pt",  # Medium - High Accuracy
        "yolov8l.pt",  # Large - Higher Accuracy
        "yolov8x.pt"   # Extra Large - Maximum Accuracy
    ]
    
    # Check custom models inside models/ directory
    custom_models = []
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith('.pt') and f not in standard_models:
                custom_models.append(f)
                
    return standard_models + custom_models
