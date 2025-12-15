"""
Enrollment service for face enrollment with quality checks.
"""
import numpy as np
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import FaceEnrollment
from app.schemas import EnrollmentResponse, EmbeddingInfo
from app.inference.face_detection import face_detector_loaded
from app.inference.face_verification import face_verifier_loaded
from app.utils.image_utils import decode_base64_image, validate_image_quality
from app.utils.vector_db import VectorDB
from app.exceptions import (
    ImageDecodeError,
    ImageQualityError,
    FaceDetectionError,
    MultipleFacesError,
    NoFaceDetectedError,
    NoValidFacesError,
    EmbeddingExtractionError
)
from app.config import settings

logger = logging.getLogger(__name__)


class EnrollmentService:
    """Service for handling face enrollment."""
    
    def __init__(self):
        """Initialize enrollment service."""
        self.face_detector = face_detector_loaded
        self.face_verifier = face_verifier_loaded
    
    async def enroll_candidate(
        self,
        db: AsyncSession,
        candidate_id: str,
        images: List[str],  # List of base64-encoded images
        metadata: Optional[Dict[str, Any]] = None
    ) -> EnrollmentResponse:
        """
        Enroll a candidate with multiple face images.
        
        Args:
            db: Database session
            candidate_id: Unique candidate identifier
            images: List of base64-encoded images
            metadata: Optional metadata
            
        Returns:
            EnrollmentResponse with embedding information
        """
        # Decode images
        decoded_images = []
        decode_errors = []
        for idx, img_base64 in enumerate(images):
            try:
                img = decode_base64_image(img_base64)
                decoded_images.append(img)
            except Exception as e:
                error_msg = f"Failed to decode image {idx + 1}: {str(e)}"
                decode_errors.append(error_msg)
                logger.warning(error_msg)
                continue
        
        if not decoded_images:
            error_detail = "No valid images could be decoded. " + (
                f"Errors: {'; '.join(decode_errors)}" if decode_errors else ""
            )
            raise ImageDecodeError(error_detail)
        
        # Process each image
        embeddings_data = []
        processing_errors = []
        
        for idx, img in enumerate(decoded_images):
            try:
                # Validate image quality
                is_valid, error_msg = validate_image_quality(
                    img,
                    min_size=(settings.min_face_size, settings.min_face_size),
                    min_sharpness=settings.min_sharpness,
                    brightness_range=(settings.min_brightness, settings.max_brightness)
                )
                
                if not is_valid:
                    error_detail = f"Image {idx + 1} quality check failed: {error_msg}"
                    processing_errors.append(error_detail)
                    logger.warning(error_detail)
                    continue
                
                # Detect faces
                detections = self.face_detector.detect(img)
                
                if len(detections) == 0:
                    error_detail = f"Image {idx + 1}: No face detected"
                    processing_errors.append(error_detail)
                    logger.warning(error_detail)
                    continue
                
                if len(detections) > 1:
                    error_detail = f"Image {idx + 1}: Multiple faces detected ({len(detections)} faces). Only one face per image is allowed."
                    processing_errors.append(error_detail)
                    logger.warning(error_detail)
                    continue
                
                detection = detections[0]

                print(detection)
                
                # Calculate quality score
                quality_metrics = self.face_detector.calculate_quality_score(
                    img, detection["bbox"]
                )
                
                # Extract embedding
                try:
                    embedding = self.face_verifier.extract_embedding_from_frame(
                        img,
                        detection["bbox"],
                        detection.get("landmarks")
                    )
                except Exception as e:
                    error_detail = f"Image {idx + 1}: Failed to extract embedding: {str(e)}"
                    processing_errors.append(error_detail)
                    logger.error(error_detail)
                    continue
                
                embeddings_data.append({
                    "embedding": embedding,
                    "quality_score": quality_metrics["overall"],
                    "sharpness_score": quality_metrics["sharpness"],
                    "brightness_score": quality_metrics["brightness"],
                    "face_bbox": detection["bbox"],
                    "landmarks": detection.get("landmarks", []),
                    "confidence": detection.get("confidence", 0.0)
                })
                
            except Exception as e:
                error_detail = f"Image {idx + 1}: Unexpected error during processing: {str(e)}"
                processing_errors.append(error_detail)
                logger.error(error_detail, exc_info=True)
                continue
        
        if not embeddings_data:
            error_detail = (
                "No valid faces detected in any of the provided images. "
                f"Processed {len(decoded_images)} images. "
            )
            if processing_errors:
                error_detail += f"Errors encountered: {'; '.join(processing_errors[:5])}"  # Show first 5 errors
                if len(processing_errors) > 5:
                    error_detail += f" (and {len(processing_errors) - 5} more errors)"
            raise NoValidFacesError(error_detail)
        
        # Select best embeddings (top 3 by quality)
        embeddings_data.sort(key=lambda x: x["quality_score"], reverse=True)
        best_embeddings = embeddings_data[:3]
        
        # Calculate average embedding
        avg_embedding = np.mean(
            [emb["embedding"] if isinstance(emb["embedding"], np.ndarray) else np.array(emb["embedding"]) 
             for emb in best_embeddings],
            axis=0
        )
        
        # Normalize average embedding
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        # Convert to vector format for storage
        avg_embedding_vector = VectorDB.numpy_to_vector(avg_embedding)
        
        # Check if enrollment already exists for this candidate
        result = await db.execute(
            select(FaceEnrollment).where(FaceEnrollment.candidate_id == candidate_id)
        )
        existing_enrollment = result.scalar_one_or_none()
        
        enrollment_id = str(uuid.uuid4())
        embedding_id = f"EMB-{candidate_id}-{uuid.uuid4().hex[:8]}"
        
        if existing_enrollment:
            # Update existing enrollment
            existing_enrollment.embedding = avg_embedding_vector
            existing_enrollment.quality_score = best_embeddings[0]["quality_score"]
            existing_enrollment.sharpness_score = best_embeddings[0]["sharpness_score"]
            existing_enrollment.brightness_score = best_embeddings[0]["brightness_score"]
            existing_enrollment.face_bbox = best_embeddings[0]["face_bbox"]
            if metadata:
                existing_enrollment.extra_metadata = metadata
            enrollment_id = str(existing_enrollment.id)
            embedding_id = existing_enrollment.embedding_id
        else:
            # Create new enrollment record
            enrollment = FaceEnrollment(
                candidate_id=candidate_id,
                embedding_id=embedding_id,
                embedding=avg_embedding_vector,
                embedding_dim=settings.embedding_dim,
                quality_score=best_embeddings[0]["quality_score"],
                sharpness_score=best_embeddings[0]["sharpness_score"],
                brightness_score=best_embeddings[0]["brightness_score"],
                face_bbox=best_embeddings[0]["face_bbox"],
                metadata=metadata
            )
            db.add(enrollment)
        
        await db.commit()
        
        # Build response
        embedding_infos = []
        for idx, emb_data in enumerate(best_embeddings):
            embedding_infos.append(EmbeddingInfo(
                embedding_id=f"{embedding_id}-{idx}",
                quality_score=emb_data["quality_score"],
                face_bbox=emb_data["face_bbox"],
                sharpness_score=emb_data["sharpness_score"],
                selected=(idx == 0)
            ))
        
        recommendations = []
        if best_embeddings[0]["quality_score"] > 0.8:
            recommendations.append("Good lighting detected")
        if best_embeddings[0]["sharpness_score"] > 0.8:
            recommendations.append("Face centered and clear")
        
        return EnrollmentResponse(
            status="success",
            candidate_id=candidate_id,
            enrollment_id=enrollment_id,
            embeddings=embedding_infos,
            average_embedding_stored=True,
            enrollment_timestamp=datetime.utcnow(),
            recommendations=recommendations if recommendations else None
        )

