"""
Session management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import json
from typing import Dict, Set
from datetime import datetime

from app.schemas import SessionStartRequest, SessionStartResponse, WSAlertMessage, AlertEvent, EventType
from app.config import settings
from app.services.session_service import SessionService
from app.services.verification_service import VerificationService
from app.utils.db import get_db
from app.models import FaceEnrollment
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove WebSocket connection."""
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to a specific connection."""
        await websocket.send_text(message)

    async def broadcast_to_session(self, session_id: str, message: str):
        """Broadcast message to all connections for a session."""
        if session_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")
                    disconnected.add(connection)

            # Remove disconnected connections
            for conn in disconnected:
                self.active_connections[session_id].discard(conn)

manager = ConnectionManager()


@router.post("/start", response_model=SessionStartResponse)
async def start_session(
    request: SessionStartRequest,
    db: AsyncSession = Depends(get_db)
) -> SessionStartResponse:
    """
    Start an exam session.

    Creates a session record and returns WebRTC configuration.
    """
    try:
        session_service = SessionService()

        # Verify candidate has enrollment
        result = await db.execute(
            select(FaceEnrollment)
            .where(FaceEnrollment.candidate_id == request.candidate_id)
            .order_by(FaceEnrollment.created_at.desc())
            .limit(1)
        )
        enrollment = result.scalar_one_or_none()

        if enrollment is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No enrollment found for candidate {request.candidate_id}"
            )

        # Start session in Redis
        config = {}
        if request.verification_policy:
            config["verification_policy"] = request.verification_policy.model_dump()
        if request.features_enabled:
            config["features_enabled"] = request.features_enabled.model_dump()
        if request.metadata:
            config.update(request.metadata)

        await session_service.start_session(
            exam_session_id=request.exam_session_id,
            candidate_id=request.candidate_id,
            config=config
        )

        # Build WebSocket URL
        websocket_url = f"ws://localhost:8000{settings.webrtc_alert_websocket_path}/{request.exam_session_id}"

        return SessionStartResponse(
            status="active",
            session_id=request.exam_session_id,
            websocket_url=websocket_url,
            calibration_required=True,
            calibration_duration_seconds=30
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start session"
        )


@router.post("/{session_id}/end", status_code=status.HTTP_204_NO_CONTENT)
async def end_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """End an exam session."""
    try:
        session_service = SessionService()

        # Check if session exists
        session = await session_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )

        # End session
        await session_service.end_session(session_id)

        # Clear frame buffer
        from app.utils.frame_buffer import FrameBuffer
        frame_buffer = FrameBuffer()
        await frame_buffer.clear_session_frames(session_id)
        await frame_buffer.close()

        # Close WebSocket connections
        await manager.broadcast_to_session(
            session_id,
            json.dumps({
                "type": "session_ended",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            })
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to end session"
        )


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get session details."""
    try:
        session_service = SessionService()
        session = await session_service.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )

        return session

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session"
        )


@router.websocket("/ws/{session_id}")
async def websocket_alerts(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time alerts.

    Connects to a session and receives alert messages.
    """
    await manager.connect(websocket, session_id)

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Send heartbeat every 10 seconds
        import asyncio
        while True:
            await asyncio.sleep(10)

            # Check if session is still active
            session_service = SessionService()
            session = await session_service.get_session(session_id)

            if not session or session.get("status") != "active":
                await websocket.send_json({
                    "type": "session_ended",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                break

            # Send heartbeat
            await websocket.send_json({
                "type": "heartbeat",
                "session_id": session_id,
                "status": "active",
                "stats": {
                    "frame_count": session.get("frame_count", 0),
                    "verification_count": session.get("verification_count", 0),
                    "last_verification": session.get("last_verification")
                },
                "timestamp": datetime.utcnow().isoformat()
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket, session_id)


async def send_alert_to_frontend(session_id: str, alert: AlertEvent):
    """
    Broadcast alert to frontend via WebSocket.

    Args:
        session_id: Exam session identifier
        alert: Alert event to send
    """
    try:
        ws_message = WSAlertMessage(
            event_id=alert.event_id,
            exam_session_id=alert.exam_session_id,
            event_type=alert.event_type,
            severity_score=alert.severity_score,
            timestamp=alert.timestamp,
            message=f"Alert: {alert.event_type.value}",
            action_required="warning" if alert.severity_score < 0.8 else "escalate",
            evidence_thumbnail=None  # Can be added if needed
        )

        message_json = ws_message.model_dump_json()
        await manager.broadcast_to_session(session_id, message_json)

    except Exception as e:
        logger.error(f"Error sending alert to frontend: {e}", exc_info=True)

