"""
Global Face Model Manager to handle single instantiation of InsightFace models.
"""
import logging
import threading
import os
from pathlib import Path
from typing import Optional, List
from insightface.app import FaceAnalysis

from app.config import settings
# from app.prediction.detectors import LSTMCheatingDetector  # Moved to lazy import

logger = logging.getLogger(__name__)


class FaceModelManager:
    """
    Singleton manager for InsightFace Analysis app.
    Ensures model is loaded only once and shared across the application.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.app: Optional[FaceAnalysis] = None
        self.predictor: Optional[any] = None  # Typed as 'any' to avoid top-level import dependency
        self.model_name = settings.face_verification_model  # Use verification model as primary
        self.device = settings.ai_model_device
        self.is_initialized = False
        self.is_predictor_initialized = False

    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_execution_providers(self) -> List[str]:
        """Get ONNX Runtime execution providers based on device."""
        if self.device == 'cuda':
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def _get_ctx_id(self) -> int:
        """Get context ID for InsightFace based on device."""
        return 0 if self.device == 'cuda' else -1

    def initialize(self):
        """
        Initialize the global FaceAnalysis model.
        This should be called at application startup.
        """
        logger.warning("initializing models")  # Changed to warning for visibility
        
        # Double-check initialization to avoid race conditions
        if self.is_initialized and self.app is not None and self.predictor is not None:
            logger.info("FaceAnalysis model and predictor already initialized")
            return

        with self._lock:
            # Re-check inside lock
            if self.is_initialized and self.app is not None and self.predictor is not None:
                return

            # 1. Initialize FaceAnalysis (InsightFace)
            try:
                logger.info(f"Initializing Global FaceAnalysis Model: {self.model_name}")
                
                pass
                # Verify paths
                root_path = settings.face_verification_model_path
                
                self.app = FaceAnalysis(
                    name=self.model_name,
                    root=root_path,
                    providers=self._get_execution_providers(),
                    # allowed_modules=['det', 'rec'] # Only load detection and recognition
                )

                # Prepare the model with detection threshold
                self.app.prepare(
                    ctx_id=self._get_ctx_id(),
                    det_thresh=settings.face_detection_confidence,
                    det_size=(settings.face_detection_size, settings.face_detection_size)
                )
                self.is_initialized = True
                logger.info("Global FaceAnalysis Model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Global FaceAnalysis Model: {e}")
                # We don't raise here to allow partial initialization if one fails (unless critical)
                # But usually FaceAnalysis is critical.
                # raise e 

            # 2. Initialize Live Proctoring Monitor (TensorFlow/LSTM)
            try:
                # Force CPU for TensorFlow if configured, BEFORE importing it
                if settings.ai_model_device == 'cpu':
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                    logger.info("Forcing TensorFlow to use CPU (CUDA_VISIBLE_DEVICES=-1)")

                # Lazy import to prevent top-level TF initialization issues
                from app.prediction.detectors import LSTMCheatingDetector

                # Initialize Live Proctoring Monitor
                BASE_DIR = Path(__file__).resolve().parent
                model_path = (
                        BASE_DIR
                        / ".."
                        / "prediction"
                        / "models"
                        / "lstm_cheating_detector"
                ).resolve()
                
                logger.info("Initializing Live Proctoring Monitor...")
                self.predictor = LSTMCheatingDetector()
                self.predictor.load(filepath=str(model_path))
                self.is_predictor_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize Live Proctoring Monitor: {e}")

    def get_app(self) -> FaceAnalysis:
        """Get the initialized FaceAnalysis app."""
        if not self.is_initialized or self.app is None:
            # Auto-initialize if not done yet (lazy loading fallback)
            logger.warning("FaceAnalysis model accessed before explicit initialization. Initializing now...")
            self.initialize()
        
        return self.app

    def get_predictor(self) -> Optional[any]:
        """Get the initialized Live Proctoring Monitor predictor."""
        if not self.is_predictor_initialized or self.predictor is None:
            # Auto-initialize if not done yet (lazy loading fallback)
            logger.warning("Live Proctoring Monitor accessed before explicit initialization. Initializing now...")
            self.initialize()

        return self.predictor
