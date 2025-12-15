"""
Global Face Model Manager to handle single instantiation of InsightFace models.
"""
import logging
import threading
from typing import Optional, List
from insightface.app import FaceAnalysis

from app.config import settings

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
        self.model_name = settings.face_verification_model  # Use verification model as primary
        self.device = settings.ai_model_device
        self.is_initialized = False

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
        if self.is_initialized and self.app is not None:
            logger.info("FaceAnalysis model already initialized")
            return

        with self._lock:
            if self.is_initialized and self.app is not None:
                return

            try:
                logger.info(f"Initializing Global FaceAnalysis Model: {self.model_name}")
                
                # Verify paths
                root_path = settings.face_verification_model_path
                
                self.app = FaceAnalysis(
                    name=self.model_name,
                    root=root_path,
                    providers=self._get_execution_providers()
                )

                # Prepare the model with detection threshold
                self.app.prepare(
                    ctx_id=self._get_ctx_id(),
                    det_thresh=settings.face_detection_confidence,
                    det_size=(640, 640)  # Standard detection size
                )

                self.is_initialized = True
                logger.info("Global FaceAnalysis Model initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize Global FaceAnalysis Model: {e}")
                raise

    def get_app(self) -> FaceAnalysis:
        """Get the initialized FaceAnalysis app."""
        if not self.is_initialized or self.app is None:
            # Auto-initialize if not done yet (lazy loading fallback)
            logger.warning("FaceAnalysis model accessed before explicit initialization. Initializing now...")
            self.initialize()
        
        return self.app
