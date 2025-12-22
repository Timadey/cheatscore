"""
Session management API endpoints.
"""
import asyncio

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
from app.services.analysis_service import AnalysisService
from app.utils.db import get_db
from app.utils.redis_client import get_redis
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

    @staticmethod
    async def send_personal_message(message: str, websocket: WebSocket):
        """Send message to a specific connection."""
        await websocket.send_text(message)

    @staticmethod
    async def receive_loop(websocket: WebSocket, session_id: str):
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                raise

            session_service = SessionService()
            session = await session_service.get_session(session_id)

            if not session or session.get("status") != "active":
                await websocket.send_json({
                    "type": "session_ended",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                break

            # Differentiate message types
            try:
                payload = None
                if isinstance(data, dict) and "extracted_features" in data:
                    payload = data["extracted_features"]
                    # TODO: verify payload  received, for now we skip verification
                elif isinstance(data, dict) and data.get("type") == "frame":
                    payload = data.get("payload") or data.get("frame")
                elif isinstance(data, dict):
                    payload = data

                if payload:
                    from app.utils.frame_buffer import FrameBuffer
                    fb = FrameBuffer()
                    await fb.store_metadata_frame(session_id, payload)
                    await fb.close()

                    # await session_service.increment_frame_count(session_id)

                logger.debug(f"Received websocket message for {session_id}")
            except Exception as e:
                logger.error(f"Error processing websocket message for {session_id}: {e}", exc_info=True)

    @staticmethod
    async def heartbeat_loop(websocket: WebSocket, session_id: str):
        while True:
            await asyncio.sleep(10)
            await websocket.send_json({
                "type": "heartbeat",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            })

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
    """End an exam session and run final analysis."""
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

        # Finalize analysis: persist analysis, dump frames, clear redis
        analysis_service = AnalysisService()
        try:
            await analysis_service.finalize_session_analysis(session_id, db)
        except Exception as e:
            logger.error(f"Final analysis failed for {session_id}: {e}", exc_info=True)

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


@router.post("/{session_id}/analyze")
async def request_analysis(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Request an on-demand analysis for a session (throttled to once per second).

    If the session is active, analysis uses frames in Redis; otherwise it will
    return persisted analysis if available.
    """
    try:
        # Throttle using Redis per-session lock of 1 second
        redis = await get_redis()
        lock_key = f"analysis_lock:{session_id}"
        acquired = await redis.set(lock_key, "1", nx=True, ex=1)
        if not acquired:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Analysis requests are limited to once per second")

        analysis_service = AnalysisService()
        report = await analysis_service.analyze_session(session_id, db=db)

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requesting analysis for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to run analysis")


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
        await asyncio.gather(
            manager.receive_loop(websocket, session_id),
            manager.heartbeat_loop(websocket, session_id),
        )

        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
