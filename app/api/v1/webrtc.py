"""
Signaling Router.
"""
import logging
from datetime import timedelta

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from livekit import api

from app.config import settings
from app.webrtc.signaling.webrtc import WebRTCManager
from app.services.session_service import SessionService

logger = logging.getLogger(__name__)

router = APIRouter()

class SessionDescription(BaseModel):
    sdp: str
    type: str

class LiveKitTokenResponse(BaseModel):
    token: str
    url: str


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


@router.get(
    "/livekit/token",
    response_model=LiveKitTokenResponse)
async def get_livekit_room_token(
    exam_id: str,
    member_id: str,
    name: str
):
    """
    Generate a LiveKit access token for a room.
    """

    if not exam_id or not member_id or not name:
        raise HTTPException(
            status_code=400,
            detail="exam id and candidate id and name are required",
        )
    from app.config import settings

    api_key = settings.livekit_api_key
    api_secret = settings.livekit_api_secret

    if not api_key or not api_secret:
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials not configured",
        )

    token = api.AccessToken(api_key, api_secret) \
        .with_identity(member_id) \
        .with_name(name) \
        .with_ttl(timedelta(hours=4)) \
        .with_grants(api.VideoGrants(
            room_join=True,
            room=exam_id,
            can_publish=True,
            can_subscribe=True
        ))

    return {"token": token.to_jwt(), "url": settings.livekit_url}