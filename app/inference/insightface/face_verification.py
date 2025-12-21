"""
Face verification module using InsightFace Python package for embedding extraction.
"""
import numpy as np
import cv2
from typing import Tuple, Optional, List
import logging

from insightface.utils import face_align

from app.inference.insightface.insightface_base import InsightFaceBase
from app.utils.face_processing_utils import FaceProcessingUtils
from app.config import settings

logger = logging.getLogger(__name__)


class FaceVerifier(InsightFaceBase):
    """Face verification using InsightFace embeddings."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize face verifier.

        Args:
            model_name: Model name (buffalo_l, buffalo_sc, etc.)
            model_path: Path to model directory
        """
        model_name = model_name or settings.face_verification_model
        super().__init__(model_name, model_path)

        self.embedding_dim = settings.embedding_dim
        self.input_size = (112, 112)  # Standard ArcFace input size

        # Initialize model
        self._load_model()

    def _load_model(self):
        """Load the face verification model using global manager."""
        super()._load_model()
        self._verify_embedding_dimension()

    def _verify_embedding_dimension(self):
        """Verify the embedding dimension of the loaded model."""
        try:
            dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
            faces = self.app.get(dummy_img)

            if faces and hasattr(faces[0], 'embedding'):
                actual_embedding_dim = len(faces[0].embedding)
                if actual_embedding_dim != self.embedding_dim:
                    logger.warning(
                        f"Model embedding dimension ({actual_embedding_dim}) doesn't "
                        f"match configured dimension ({self.embedding_dim}). "
                        "Using model dimension."
                    )
                    self.embedding_dim = actual_embedding_dim
        except Exception as e:
            logger.info(f"Could not verify embedding dimension: {e}")

    def extract_embedding_from_frame(
        self,
        frame: np.ndarray,
        bbox: List[int],
    ) -> np.ndarray:
        """
        Extract embedding from a frame given bounding box and landmarks.

        This is the RECOMMENDED method for embedding extraction.

        Args:
            frame: Full frame image (BGR format)
            bbox: Bounding box [x1, y1, x2, y2] of the target face
            # landmarks: Optional facial landmarks (not used in this implementation)

        Returns:
            Face embedding vector
        """
        self._verify_model_loaded()

        try:
            # Convert BGR to RGB - REMOVED (InsightFace expects BGR)
            # frame_rgb = FaceProcessingUtils.convert_bgr_to_rgb(frame)

            # Get all faces with embeddings
            faces = self.app.get(frame)

            if not faces:
                raise RuntimeError("No face detected in frame")

            # Find matching face
            if len(faces) == 1:
                face = faces[0]
            else:
                face = self.utils.find_matching_face(faces, bbox)
                if face is None:
                    raise RuntimeError(
                        f"Could not match any detected face to target bbox. "
                        f"Detected {len(faces)} faces, target bbox: {bbox}"
                    )

            # Verify embedding exists
            if not hasattr(face, 'embedding') or face.embedding is None:
                raise RuntimeError("Face embedding not available")

            # Normalize and return embedding
            embedding = FaceProcessingUtils.normalize_embedding(face.embedding)

            logger.debug(f"Extracted embedding with dimension {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction from frame error: {e}", exc_info=True)
            raise

    def extract_embedding_direct(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding directly from a frame using InsightFace's detection + recognition.

        Convenience method that combines detection and embedding extraction.

        Args:
            frame: Full frame image (BGR format)

        Returns:
            Face embedding vector for the first detected face, or None if no face detected
        """
        self._verify_model_loaded()

        try:
            # Convert BGR to RGB - REMOVED (InsightFace expects BGR)
            # frame_rgb = FaceProcessingUtils.convert_bgr_to_rgb(frame)

            # Get faces with embeddings
            faces = self.app.get(frame)

            if not faces:
                logger.info("No face detected in frame")
                return None

            # Return embedding from first face
            embedding = FaceProcessingUtils.normalize_embedding(faces[0].embedding)

            return embedding

        except Exception as e:
            logger.error(f"Direct embedding extraction error: {e}", exc_info=True)
            return None

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector (should be L2-normalized)
            emb2: Second embedding vector (should be L2-normalized)

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        return FaceProcessingUtils.compute_cosine_similarity(emb1, emb2)

    def verify(
        self,
        current_embedding: np.ndarray,
        enrolled_embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Verify if face matches enrolled embedding.

        Args:
            current_embedding: Aligned face image embedding
            enrolled_embedding: Stored enrollment embedding
            threshold: Similarity threshold (default from settings)

        Returns:
            Tuple of (is_match, similarity_score)
        """
        threshold = threshold or settings.face_match_threshold

        # Extract embedding from face crop
        # current_embedding = self.extract_embedding(face_crop)

        # Compute similarity
        similarity = self.compute_similarity(current_embedding, enrolled_embedding)

        # Check against threshold
        is_match = similarity >= threshold

        return is_match, similarity

face_verifier_loaded = FaceVerifier()