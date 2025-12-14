"""
Common utilities for face processing operations.
"""
import numpy as np
import cv2
import base64
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FaceProcessingUtils:
    """Utility functions for face processing."""

    @staticmethod
    def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute Intersection over Union between two bounding boxes.

        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]

        Returns:
            IoU score (0.0 to 1.0)
        """
        # Get intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    @staticmethod
    def find_matching_face(
        faces: List,
        target_bbox: List[int],
        iou_threshold: float = 0.5
    ):
        """
        Find face that best matches the target bounding box.

        Args:
            faces: List of Face objects from InsightFace
            target_bbox: Target bbox [x1, y1, x2, y2]
            iou_threshold: Minimum IoU to consider a match

        Returns:
            Best matching Face object, or None if no good match
        """
        best_face = None
        best_iou = 0

        for face in faces:
            iou = FaceProcessingUtils.compute_iou(
                face.bbox.tolist(),
                target_bbox
            )
            if iou > best_iou:
                best_iou = iou
                best_face = face

        return best_face if best_iou >= iou_threshold else None

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        L2-normalize an embedding vector.

        Args:
            embedding: Embedding vector

        Returns:
            L2-normalized embedding
        """
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    @staticmethod
    def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Ensure embeddings are normalized
        emb1_norm = FaceProcessingUtils.normalize_embedding(emb1)
        emb2_norm = FaceProcessingUtils.normalize_embedding(emb2)

        # Cosine similarity (dot product of normalized vectors)
        similarity = np.dot(emb1_norm, emb2_norm)

        # Clamp to [0, 1] range
        similarity = max(0.0, min(1.0, float(similarity)))

        return similarity

    @staticmethod
    def decode_base64_frame(frame_data: str) -> bytes:
        """
        Decode base64 encoded frame data.

        Args:
            frame_data: Base64 encoded frame string

        Returns:
            Decoded frame bytes
        """
        return base64.b64decode(frame_data)

    @staticmethod
    def convert_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR frame to RGB.

        Args:
            frame: BGR frame

        Returns:
            RGB frame
        """
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    @staticmethod
    def calculate_face_quality_score(
        frame: np.ndarray,
        bbox: List[int],
        min_face_size: int = 80
    ) -> dict:
        """
        Calculate quality metrics for a detected face.

        Args:
            frame: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            min_face_size: Minimum acceptable face size

        Returns:
            Dictionary with quality metrics
        """
        x1, y1, x2, y2 = bbox

        # Ensure valid bounding box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return {
                "sharpness": 0.0,
                "brightness": 0.0,
                "size_score": 0.0,
                "overall": 0.0
            }

        # Sharpness (Laplacian variance)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 100.0, 1.0)  # Normalize to 0-1

        # Brightness
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0

        # Size score
        face_area = (x2 - x1) * (y2 - y1)
        min_area = min_face_size ** 2
        size_score = min(face_area / min_area, 1.0)

        # Overall quality
        overall = (sharpness * 0.4 + brightness_score * 0.3 + size_score * 0.3)

        return {
            "sharpness": float(sharpness),
            "brightness": float(brightness_score),
            "size_score": float(size_score),
            "overall": float(overall)
        }

    @staticmethod
    def simple_face_crop_and_resize(
        frame: np.ndarray,
        target_size: tuple = (112, 112)
    ) -> np.ndarray:
        """
        Simple center crop and resize for face images.

        Args:
            frame: Input image
            target_size: Target size (width, height)

        Returns:
            Cropped and resized face
        """
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_size = min(w, h)
        x1 = max(0, center_x - crop_size // 2)
        y1 = max(0, center_y - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        face_crop = frame[y1:y2, x1:x2]
        return cv2.resize(face_crop, target_size)