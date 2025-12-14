"""
Enrollment API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.schemas import EnrollmentRequest, EnrollmentResponse
from app.services.enrollment_service import EnrollmentService
from app.exceptions import (
    EnrollmentError,
    EmbeddingExtractionError
)
from app.utils.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=EnrollmentResponse, status_code=status.HTTP_201_CREATED)
async def enroll_candidate(
    request: EnrollmentRequest,
    db: AsyncSession = Depends(get_db)
) -> EnrollmentResponse:
    """
    Enroll a candidate with face images.
    
    Accepts 1-10 images, processes them, and stores the best embeddings.
    """
    try:
        service = EnrollmentService()
        
        # Extract image base64 strings
        images = [img.image_base64 for img in request.images]
        
        response = await service.enroll_candidate(
            db=db,
            candidate_id=request.candidate_id,
            images=images,
            metadata=request.metadata
        )
        
        return response
        
    except EmbeddingExtractionError as e:
        logger.error(f"Embedding extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract face embedding: {str(e)}"
        )
    except EnrollmentError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected enrollment error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enroll candidate"
        )


@router.get("/{candidate_id}", response_model=EnrollmentResponse)
async def get_enrollment(
    candidate_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get enrollment details for a candidate.
    """
    from app.models import FaceEnrollment
    from sqlalchemy import select
    from app.schemas import EmbeddingInfo
    
    try:
        result = await db.execute(
            select(FaceEnrollment)
            .where(FaceEnrollment.candidate_id == candidate_id)
            .order_by(FaceEnrollment.created_at.desc())
            .limit(1)
        )
        enrollment = result.scalar_one_or_none()
        
        if enrollment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No enrollment found for candidate {candidate_id}"
            )
        
        return EnrollmentResponse(
            status="success",
            candidate_id=candidate_id,
            enrollment_id=str(enrollment.id),
            embeddings=[
                EmbeddingInfo(
                    embedding_id=enrollment.embedding_id,
                    quality_score=enrollment.quality_score,
                    face_bbox=enrollment.face_bbox,
                    sharpness_score=enrollment.sharpness_score,
                    selected=True
                )
            ],
            average_embedding_stored=True,
            enrollment_timestamp=enrollment.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get enrollment error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get enrollment"
        )


@router.delete("/{candidate_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_enrollment(
    candidate_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete enrollment for a candidate.
    """
    from app.models import FaceEnrollment
    from sqlalchemy import select
    
    try:
        result = await db.execute(
            select(FaceEnrollment).where(FaceEnrollment.candidate_id == candidate_id)
        )
        enrollments = result.scalars().all()
        
        if not enrollments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No enrollment found for candidate {candidate_id}"
            )
        
        for enrollment in enrollments:
            await db.delete(enrollment)
        
        await db.commit()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete enrollment error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete enrollment"
        )

