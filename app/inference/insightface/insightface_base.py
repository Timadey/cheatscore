"""
Base class for InsightFace models with shared initialization logic.
"""
import numpy as np
import logging
from typing import Optional, List
from insightface.app import FaceAnalysis

from app.config import settings
from app.utils.face_processing_utils import FaceProcessingUtils

logger = logging.getLogger(__name__)


class InsightFaceBase:
    """Base class for InsightFace models with common initialization."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize InsightFace base.

        Args:
            model_name: Model name (buffalo_l, buffalo_sc, scrfd_500m_bnkps, etc.)
            model_path: Path to model directory
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.model_path = model_path or settings.face_detection_model_path
        self.device = device or settings.ai_model_device
        self.app: Optional[FaceAnalysis] = None
        self.utils = FaceProcessingUtils

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

    def _verify_model_loaded(self):
        """Verify that the model is loaded."""
        if self.app is None:
            raise RuntimeError(
                f"{self.__class__.__name__} model not loaded. "
                "Cannot perform inference."
            )