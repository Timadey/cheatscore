"""
Face detection module using InsightFace Python package.
"""
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
import logging

from app.inference.insightface_base import InsightFaceBase
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

    def _load_model(self):
        """Load the face detection model using global manager."""
        try:
            # Use global model manager
            from app.inference.face_model_manager import FaceModelManager
            self.app = FaceModelManager.get_instance().get_app()
        except Exception as e:
            logger.error(f"Failed to load InsightFace model from manager: {e}")
            # Fall back to OpenCV if InsightFace fails
            logger.warning("Falling back to OpenCV face detection")
            self.app = None

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
        if self.app is None:
            # Fallback to OpenCV DNN face detector if model not loaded
            return self._detect_opencv_fallback(frame)

        try:
            # InsightFace expects BGR format (no conversion needed)
            img_rgb = FaceProcessingUtils.convert_bgr_to_rgb(frame)

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
            return self._detect_opencv_fallback(frame)

    def _detect_opencv_fallback(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fallback face detection using OpenCV DNN or Haar Cascade.

        Args:
            frame: Input image

        Returns:
            List of detections
        """
        # Try OpenCV DNN first
        try:
            prototxt_path = "models/opencv_face_detector.pbtxt"
            model_path = "models/opencv_face_detector_uint8.pb"

            net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                1.0,
                (300, 300),
                [104, 117, 123]
            )
            net.setInput(blob)
            detections = net.forward()

            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)

                    faces.append({
                        "bbox": [x1, y1, x2, y2],
                        "landmarks": [],
                        "confidence": float(confidence)
                    })

            return faces

        except Exception as e:
            logger.warning(f"OpenCV DNN detection failed: {e}, using Haar Cascade")
            return self._detect_haar_cascade(frame)

    def _detect_haar_cascade(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback to Haar Cascade detector."""
        try:
            cascade_path = (
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            face_cascade = cv2.CascadeClassifier(cascade_path)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rect = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            detections = []
            for (x, y, w, h) in faces_rect:
                detections.append({
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "landmarks": [],
                    "confidence": 0.9  # Default confidence
                })

            return detections

        except Exception as e:
            logger.error(f"Haar Cascade detection failed: {e}")
            return []

    def detect_batch(
        self,
        frames: List[np.ndarray]
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect faces in a batch of frames.

        Note: InsightFace processes frames individually by default.

        Args:
            frames: List of input images

        Returns:
            List of detection lists (one per frame)
        """
        if self.app is None:
            # Fallback to sequential detection
            return [self.detect(frame) for frame in frames]

        try:
            # Process each frame
            all_detections = []
            for frame in frames:
                detections = self.detect(frame)
                all_detections.append(detections)

            return all_detections

        except Exception as e:
            logger.error(f"Batch detection error: {e}, falling back to sequential")
            return [self.detect(frame) for frame in frames]

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