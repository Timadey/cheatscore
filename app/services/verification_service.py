"""
Verification service for face verification.
"""
import numpy as np
import cv2
from typing import Optional
from datetime import datetime
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import FaceEnrollment
from app.schemas import VerificationResponse, FaceQuality
from app.inference.insightface.face_detection import face_detector_loaded
from app.inference.insightface.face_verification import face_verifier_loaded
from app.utils.image_utils import decode_base64_image
from app.exceptions import NoEnrollmentError
from app.config import settings

logger = logging.getLogger(__name__)


class VerificationService:
    """Service for handling face verification."""
    
    def __init__(self):
        """Initialize verification service."""
        self.face_detector = face_detector_loaded
        self.face_verifier = face_verifier_loaded
    
    async def verify_frame(
        self,
        db: AsyncSession,
        candidate_id: str,
        image_base64: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
        threshold: Optional[float] = None
    ) -> VerificationResponse:
        """
        Verify a single frame against enrolled embedding.
        
        Args:
            db: Database session
            candidate_id: Candidate identifier
            image_base64: Base64-encoded image (if frame not provided)
            frame: Pre-decoded image array (if image_base64 not provided)
            threshold: Similarity threshold (default from settings)
            
        Returns:
            VerificationResponse with match result
        """
        # Get image
        if frame is None:
            if image_base64 is None:
                raise ValueError("Either image_base64 or frame must be provided")
            frame = decode_base64_image(image_base64)
        
        # Get enrolled embedding
        result = await db.execute(
            select(FaceEnrollment)
            .where(FaceEnrollment.candidate_id == candidate_id)
            .order_by(FaceEnrollment.created_at.desc())
            .limit(1)
        )
        enrollment = result.scalar_one_or_none()
        
        if enrollment is None:
            raise NoEnrollmentError(f"No enrollment found for candidate {candidate_id}")
        
        # Convert vector to numpy array
        enrolled_embedding = enrollment.embedding
        # enrolled_embedding = np.array(enrollment.embedding, dtype=np.float32)
        # enrolled_embedding = VectorDB.vector_to_numpy(enrollment.embedding)
        
        # Detect face
        detections = self.face_detector.detect(frame)
        
        if len(detections) == 0:
            raise ValueError("No face detected in image")
        
        if len(detections) > 1:
            raise ValueError("Multiple face detected in image")

        detection = detections[0]
        
        # Extract embedding from frame
        current_embedding = self.face_verifier.extract_embedding_from_frame(
            frame,
            detection["bbox"],
            # detection.get("landmarks")
        )
        
        # # Extract face crop for verification
        # x1, y1, x2, y2 = detection["bbox"]
        # face_crop = frame[y1:y2, x1:x2]
        #
        # # Align face if landmarks available
        # if detection.get("landmarks") and len(detection["landmarks"]) >= 5:
        #     face_crop = self.face_verifier.align_face(frame, detection["landmarks"])
        # else:
        #     face_crop = cv2.resize(face_crop, self.face_verifier.input_size)
        #
        # Verify
        threshold = threshold or settings.face_match_threshold
        is_match, similarity = self.face_verifier.verify(
            current_embedding,
            enrolled_embedding,
            threshold
        )
        
        # Calculate quality metrics
        quality_metrics = self.face_detector.calculate_quality_score(
            frame, detection["bbox"]
        )
        
        # Calculate pose deviation (simplified)
        pose_deviation = 0.0  # TODO: Calculate actual pose deviation
        
        face_quality = FaceQuality(
            sharpness=quality_metrics["sharpness"],
            pose_deviation=pose_deviation,
            lighting_quality=quality_metrics["brightness"]
        )
        
        return VerificationResponse(
            match=is_match,
            similarity_score=similarity,
            threshold_used=threshold,
            comparison_embedding_id=enrollment.embedding_id,
            face_quality=face_quality,
            timestamp=datetime.utcnow()
        )

