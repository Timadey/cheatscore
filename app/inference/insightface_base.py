"""
Base class for InsightFace models with shared initialization logic.
"""
import numpy as np
import logging
from typing import Optional, List
from insightface.app import FaceAnalysis

from app.config import settings

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

    def _get_execution_providers(self) -> List[str]:
        """
        Get ONNX Runtime execution providers based on device.

        Returns:
            List of execution providers
        """
        if self.device == 'cuda':
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def _get_ctx_id(self) -> int:
        """
        Get context ID for InsightFace based on device.

        Returns:
            Context ID (-1 for CPU, 0+ for GPU)
        """
        return 0 if self.device == 'cuda' else -1

    def _load_model(
        self,
        det_thresh: Optional[float] = None,
        det_size: tuple = (640, 640)
    ):
        """
        Load InsightFace model.

        Args:
            det_thresh: Detection threshold
            det_size: Detection input size
        """
        try:
            logger.info(f"Loading InsightFace model: {self.model_name}")

            # Initialize FaceAnalysis app
            self.app = FaceAnalysis(
                name=self.model_name,
                root=self.model_path,
                providers=self._get_execution_providers()
            )

            # Prepare the model
            self.app.prepare(
                ctx_id=self._get_ctx_id(),
                det_thresh=det_thresh or settings.face_detection_confidence,
                det_size=det_size
            )

            logger.info(
                f"InsightFace model '{self.model_name}' loaded successfully. "
                f"Device: {self.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise

    def _verify_model_loaded(self):
        """Verify that the model is loaded."""
        if self.app is None:
            raise RuntimeError(
                f"{self.__class__.__name__} model not loaded. "
                "Cannot perform inference."
            )