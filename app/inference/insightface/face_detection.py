"""
Face detection module using InsightFace Python package.
"""
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
import logging

from app.inference.insightface.insightface_base import InsightFaceBase
from app.utils.face_processing_utils import FaceProcessingUtils
from app.config import settings

logger = logging.getLogger(__name__)


class FaceDetector(InsightFaceBase):
    """Face detection using InsightFace package."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize face detector.

        Args:
            model_name: Model name (scrfd_500m_bnkps, retinaface_r50_v1, etc.)
            model_path: Path to model directory
        """
        model_name = model_name or settings.face_detection_model
        super().__init__(model_name, model_path)

        self.confidence_threshold = settings.face_detection_confidence

        # Initialize model
        self._load_model()

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a single frame.

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            List of detections, each containing:
            - bbox: [x1, y1, x2, y2] bounding box
            - landmarks: 5 facial landmarks [[x1,y1], [x2,y2], ...]
            - confidence: Detection confidence score
        """
        self._verify_model_loaded()

        try:
            # InsightFace expects BGR format (no conversion needed)
            img_rgb = self.utils.convert_bgr_to_rgb(frame)

            # Get face detections
            faces = self.app.get(img_rgb)

            # Convert to our standard format
            detections = []
            for face in faces:
                # Extract bounding box
                bbox = face.bbox.astype(int).tolist()

                # Extract landmarks (keypoints)
                landmarks = (
                    face.kps.tolist()
                    if hasattr(face, 'kps') and face.kps is not None
                    else []
                )

                # Extract confidence
                confidence = (
                    float(face.det_score)
                    if hasattr(face, 'det_score')
                    else 0.9
                )

                # Filter by confidence threshold
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "bbox": bbox,
                        "landmarks": landmarks,
                        "confidence": confidence,
                        "face": face
                    })

            return detections

        except Exception as e:
            logger.error(f"Face detection inference error: {e}", exc_info=True)
            # Fallback to OpenCV on error
            raise RuntimeError("Face detection inference failed")

    def calculate_quality_score(
        self,
        frame: np.ndarray,
        bbox: List[int]
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for a detected face.

        Args:
            frame: Input image
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Dictionary with quality metrics
        """
        return FaceProcessingUtils.calculate_face_quality_score(
            frame,
            bbox,
            min_face_size=settings.min_face_size
        )

face_detector_loaded = FaceDetector()