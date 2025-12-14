"""
Verification API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.schemas import VerificationRequest, VerificationResponse
from app.services.verification_service import VerificationService
from app.exceptions import (
    NoEnrollmentError,
    NoFaceDetectedError,
    FaceDetectionError,
    EmbeddingExtractionError,
    VerificationError
)
from app.utils.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=VerificationResponse)
async def verify_face(
    request: VerificationRequest,
    db: AsyncSession = Depends(get_db)
) -> VerificationResponse:
    """
    Verify a face image against enrolled embedding.
    
    Can be used for ad-hoc verification or during exam sessions.
    """
    try:
        service = VerificationService()
        
        response = await service.verify_frame(
            db=db,
            candidate_id=request.candidate_id,
            image_base64=request.image_base64,
            threshold=None  # Use default from settings
        )
        
        return response
        
    except NoEnrollmentError as e:
        logger.error(f"No enrollment error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except NoFaceDetectedError as e:
        logger.error(f"No face detected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No face detected in image: {str(e)}"
        )
    except FaceDetectionError as e:
        logger.error(f"Face detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face detection failed: {str(e)}"
        )
    except EmbeddingExtractionError as e:
        logger.error(f"Embedding extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract face embedding: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Verification validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected verification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify face"
        )

