"""
Smart Crowd AI: AI Detection Engine
Supports YOLOv8 with graceful fallback to OpenCV HOG People Detector and Synthetic blob estimation.
Generates bounding boxes, centroids, and Gaussian density heatmap overlays.
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Any, Optional

# Attempt importing ultralytics YOLO
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class CrowdDetector:
    """Manages AI detection models, fallback cascades, and visualization rendering."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.35):
        self.model_name = model_name
        self.confidence = confidence
        self.yolo_model = None
        self.hog_detector = None
        self.active_engine = "Uninitialized"
        self._init_models()

    def _init_models(self):
        """Initializes YOLOv8 if available, or falls back to OpenCV HOG."""
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(self.model_name)
                self.active_engine = "YOLOv8-Nano (PyTorch Deep Learning)"
                return
            except Exception as e:
                print(f"[Detector] YOLO init note: {e}. Falling back to OpenCV HOG.")

        try:
            self.hog_detector = cv2.HOGDescriptor()
            self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.active_engine = "OpenCV HOG Pedestrian Classifier (CPU Native)"
        except Exception as e:
            print(f"[Detector] HOG init note: {e}. Fallback to Synthetic Engine.")
            self.active_engine = "Synthetic Heuristic Model"

    def detect(
        self,
        image_bgr: np.ndarray,
        engine_override: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detects persons in image_bgr.
        Supports automatic model execution, fallback, and synthetic sample detection.
        """
        start_time = time.time()
        h, w = image_bgr.shape[:2]
        boxes = []
        centroids = []
        conf = conf_threshold if conf_threshold is not None else self.confidence
        engine_used = self.active_engine

        # Check if caller requested a specific engine
        if engine_override == "OpenCV HOG Pedestrian Classifier":
            boxes, centroids, engine_used = self._run_hog(image_bgr)
        elif engine_override == "Synthetic / Blob Detection":
            boxes, centroids, engine_used = self._run_synthetic_silhouettes(image_bgr)
        else:
            # Default / Auto: Try YOLOv8 first
            yolo_ran_successfully = False
            if self.yolo_model is not None:
                try:
                    results = self.yolo_model(image_bgr, classes=[0], conf=conf, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            coords = box.xyxy[0].cpu().numpy().astype(int)
                            c = float(box.conf[0].cpu().numpy())
                            x1, y1, x2, y2 = coords
                            boxes.append([int(x1), int(y1), int(x2), int(y2), round(c, 2)])
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            centroids.append((cx, cy))
                    yolo_ran_successfully = True
                    engine_used = "YOLOv8-Nano (COCO AI)"
                except Exception:
                    yolo_ran_successfully = False

            # If YOLO returned 0 and it's a synthetic sample image or YOLO was unavailable
            if (not yolo_ran_successfully or len(boxes) == 0):
                # Check if it has synthetic markers or if HOG detects anything
                syn_boxes, syn_centroids, syn_engine = self._run_synthetic_silhouettes(image_bgr)
                if len(syn_boxes) > 0 and len(boxes) == 0:
                    boxes = syn_boxes
                    centroids = syn_centroids
                    engine_used = "Synthetic Pedestrian Analyzer (Demo Sample)"
                elif not yolo_ran_successfully:
                    boxes, centroids, engine_used = self._run_hog(image_bgr)

        inference_time_ms = round((time.time() - start_time) * 1000, 1)

        # Calculate crowd spatial clustering metric
        cluster_factor = 0.5
        if len(centroids) > 3:
            pts = np.array(centroids)
            std_x = np.std(pts[:, 0]) / (w + 1e-5)
            std_y = np.std(pts[:, 1]) / (h + 1e-5)
            spread = (std_x + std_y) / 2.0
            cluster_factor = round(float(np.clip(1.0 - (spread * 2.5), 0.1, 0.95)), 2)

        is_sim = "Synthetic" in engine_used

        return {
            "person_count": len(boxes),
            "boxes": boxes,
            "centroids": centroids,
            "engine_used": engine_used,
            "inference_time_ms": inference_time_ms,
            "cluster_factor": cluster_factor,
            "is_simulated": is_sim
        }

    def _run_hog(self, image_bgr: np.ndarray) -> Tuple[List[List[Any]], List[Tuple[int, int]], str]:
        """Runs OpenCV HOG Pedestrian detection."""
        if self.hog_detector is None:
            self.hog_detector = cv2.HOGDescriptor()
            self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        boxes = []
        centroids = []
        h, w = image_bgr.shape[:2]
        scale = 1.0
        if w > 800:
            scale = 800.0 / w
            resized = cv2.resize(image_bgr, (800, int(h * scale)))
        else:
            resized = image_bgr

        rects, weights = self.hog_detector.detectMultiScale(
            resized, winStride=(4, 4), padding=(8, 8), scale=1.05
        )

        for (rx, ry, rw, rh), weight in zip(rects, weights):
            x1 = int(rx / scale)
            y1 = int(ry / scale)
            x2 = int((rx + rw) / scale)
            y2 = int((ry + rh) / scale)
            conf = min(0.99, max(0.40, float(weight)))
            boxes.append([x1, y1, x2, y2, round(conf, 2)])
            centroids.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

        return boxes, centroids, "OpenCV HOG Pedestrian Classifier"

    def _run_synthetic_silhouettes(self, image_bgr: np.ndarray) -> Tuple[List[List[Any]], List[Tuple[int, int]], str]:
        """Detects silhouettes from synthetic crowd scenes using morphological analysis."""
        boxes = []
        centroids = []
        h, w = image_bgr.shape[:2]

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Background floor is around 38-42
        diff = cv2.absdiff(gray, 40)
        _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)

        # Ignore the top gate barrier banner and bottom watermark
        mask = np.zeros_like(thresh)
        mask[55:h - 45, 35:w - 35] = 255
        thresh = cv2.bitwise_and(thresh, mask)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if 60 < area < 12000:
                x, y, bw, bh = cv2.boundingRect(c)
                # Filter noise
                if bw < 10 or bh < 15:
                    continue
                # If a large contour contains multiple clustered people, split proportionally
                estimated_people = max(1, int(round(area / 450.0)))
                if estimated_people > 1:
                    step_w = bw / estimated_people
                    for p in range(estimated_people):
                        px1 = int(x + p * step_w)
                        px2 = int(px1 + step_w)
                        boxes.append([px1, y, px2, y + bh, 0.88])
                        centroids.append((int((px1 + px2) / 2), int(y + bh / 2)))
                else:
                    boxes.append([x, y, x + bw, y + bh, 0.92])
                    centroids.append((int(x + bw / 2), int(y + bh / 2)))

        return boxes, centroids, "Synthetic Crowd Silhouette Detector"

    def annotate(
        self,
        image_bgr: np.ndarray,
        boxes: List[List[Any]],
        centroids: List[Tuple[int, int]],
        risk_info: Dict[str, Any],
        show_boxes: bool = True,
        show_centroids: bool = True,
        show_heatmap: bool = True,
        heatmap_opacity: float = 0.45
    ) -> np.ndarray:
        """
        Draws annotated visualization over image_bgr.
        Creates bounding boxes, centroid dots, Gaussian density heatmap, and status HUD.
        """
        output = image_bgr.copy()
        h, w = output.shape[:2]

        risk_level = risk_info.get("risk_level", "SAFE")
        if risk_level == "STAMPEDE RISK":
            theme_color_bgr = (40, 40, 239)   # Red
        elif risk_level == "CRITICAL":
            theme_color_bgr = (22, 115, 249)  # Orange
        elif risk_level == "WARNING":
            theme_color_bgr = (11, 158, 245)  # Amber
        elif risk_level == "MODERATE":
            theme_color_bgr = (246, 130, 59)  # Blue
        else:
            theme_color_bgr = (129, 185, 16)  # Green

        # 1. Generate and blend Gaussian Density Heatmap
        if show_heatmap and len(centroids) > 0:
            heat_canvas = np.zeros((h, w), dtype=np.float32)
            kernel_radius = max(25, int(min(h, w) * 0.08))

            for (cx, cy) in centroids:
                cv2.circle(heat_canvas, (cx, cy), kernel_radius, 1.0, -1)

            blur_size = max(31, int(kernel_radius * 2) | 1)
            heat_blurred = cv2.GaussianBlur(heat_canvas, (blur_size, blur_size), 0)

            max_val = np.max(heat_blurred)
            if max_val > 0:
                heat_norm = (heat_blurred / max_val * 255.0).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

                mask = (heat_norm > 15).astype(np.uint8)
                mask_3c = cv2.merge([mask, mask, mask])
                blended = cv2.addWeighted(output, 1.0 - heatmap_opacity, heat_color, heatmap_opacity, 0)
                output = np.where(mask_3c == 1, blended, output)

        # 2. Draw Bounding Boxes
        if show_boxes:
            for box in boxes:
                x1, y1, x2, y2, conf = box
                cv2.rectangle(output, (x1, y1), (x2, y2), theme_color_bgr, 2)
                label = f"Person {int(conf * 100)}%"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(output, (x1, max(0, y1 - 18)), (x1 + lw + 6, max(0, y1)), theme_color_bgr, -1)
                cv2.putText(output, label, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Centroids
        if show_centroids:
            for (cx, cy) in centroids:
                cv2.circle(output, (cx, cy), 4, (0, 255, 255), -1)
                cv2.circle(output, (cx, cy), 6, (0, 0, 0), 1)

        # 4. Draw HUD / Overlay Banner at top
        banner_h = 50
        cv2.rectangle(output, (0, 0), (w, banner_h), (20, 24, 33), -1)
        cv2.line(output, (0, banner_h), (w, banner_h), theme_color_bgr, 2)

        count = len(boxes)
        density = risk_info.get("density", 0.0)
        risk_score = risk_info.get("risk_score", 0.0)
        fruin = risk_info.get("fruin_los", "LOS A")

        hud_text_left = f"Smart Crowd AI | Count: {count} persons | Density: {density} p/m2"
        hud_text_right = f"Status: {risk_level} ({risk_score}/100) | {fruin}"

        cv2.putText(output, hud_text_left, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        (rw, _), _ = cv2.getTextSize(hud_text_right, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.putText(output, hud_text_right, (max(w - rw - 15, 15), 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, theme_color_bgr, 2, cv2.LINE_AA)

        return output
