"""
Signaling Router.
"""
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.signaling.webrtc import WebRTCManager
from app.services.session_service import SessionService

logger = logging.getLogger(__name__)

router = APIRouter()

class SessionDescription(BaseModel):
    sdp: str
    type: str

class OfferRequest(BaseModel):
    sdp: str
    type: str
    exam_session_id: str
    candidate_id: str

@router.post("/offer", response_model=SessionDescription)
async def offer(request: OfferRequest):
    """
    Handle WebRTC SDP Offer.
    """
    # Verify session
    session_service = SessionService()
    if not await session_service.is_session_active(request.exam_session_id):
        raise HTTPException(status_code=400, detail="Session not active")

    manager = WebRTCManager.get_instance()
    
    try:
        sdp_answer = await manager.create_session(
            request.exam_session_id,
            request.candidate_id,
            request.sdp,
            request.type
        )
        return {"sdp": sdp_answer, "type": "answer"}
    except Exception as e:
        logger.error(f"Error handling offer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
