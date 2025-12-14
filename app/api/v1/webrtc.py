"""
WebRTC frame ingestion API endpoints.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from typing import Optional
from datetime import datetime

from app.webrtc.gateway_adapter import WebRTCGatewayAdapter
from app.services.session_service import SessionService
from app.dispatcher.alert_dispatcher import AlertDispatcher
from app.utils.websocket_utils import WebSocketFrameHandler
from app.utils.face_processing_utils import FaceProcessingUtils
from app.utils.db import get_db, AsyncSessionLocal
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/frames/{exam_session_id}")
async def receive_webrtc_frames(
    websocket: WebSocket,
    exam_session_id: str,
    candidate_id: Optional[str] = None
):
    """
    WebSocket endpoint for receiving WebRTC video frames from frontend.

    Expected message format:
    {
        "type": "frame",
        "candidate_id": "CAND-001",
        "frame_data": "base64_encoded_frame",
        "timestamp": "2024-01-01T00:00:00Z",
        "metadata": {}
    }
    """
    await websocket.accept()

    session_service = SessionService()
    gateway_adapter = WebRTCGatewayAdapter()
    alert_dispatcher = AlertDispatcher()
    ws_handler = WebSocketFrameHandler()

    # Get database session
    db = AsyncSessionLocal()

    try:
        # Verify session is active
        if not await session_service.is_session_active(exam_session_id):
            await ws_handler.send_error(
                websocket,
                f"Session {exam_session_id} is not active"
            )
            return

        # Get candidate_id from session if not provided
        if not candidate_id:
            candidate_id = await session_service.get_candidate_id(exam_session_id)
            if not candidate_id:
                await ws_handler.send_error(websocket, "Candidate ID not found")
                return

        # Send connection confirmation
        await ws_handler.send_connection_confirmation(
            websocket,
            exam_session_id,
            candidate_id
        )

        # Create frame stream generator
        frame_stream = ws_handler.create_frame_stream_generator(websocket)

        # Process frame stream
        frame_metadata = {
            "source": "webrtc_websocket",
            "candidate_id": candidate_id
        }

        async for alert in gateway_adapter.receive_frame_stream(
            exam_session_id=exam_session_id,
            candidate_id=candidate_id,
            frame_stream=frame_stream,
            frame_metadata=frame_metadata,
            db_session=db
        ):
            if alert:
                # Send alert to WebSocket client
                await ws_handler.send_alert(websocket, alert)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {exam_session_id}")
    except Exception as e:
        logger.error(f"Error in WebRTC frame reception: {e}", exc_info=True)
        try:
            await ws_handler.send_error(websocket, str(e))
        except:
            pass
    finally:
        await gateway_adapter.close()
        try:
            await db.close()
        except:
            pass


@router.post("/frames/{exam_session_id}/upload")
async def upload_frame_batch(
    exam_session_id: str,
    frames: list[dict],
    db: AsyncSession = Depends(get_db)
):
    """
    HTTP endpoint for batch frame upload (alternative to WebSocket).

    Request body:
    {
        "candidate_id": "CAND-001",
        "frames": [
            {
                "frame_data": "base64_encoded_frame",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        ]
    }
    """
    try:
        session_service = SessionService()

        # Verify session is active
        if not await session_service.is_session_active(exam_session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Session {exam_session_id} is not active"
            )

        # Get candidate ID
        candidate_id = await session_service.get_candidate_id(exam_session_id)
        if not candidate_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Candidate ID not found"
            )

        gateway_adapter = WebRTCGatewayAdapter()
        alerts = []

        # Process each frame
        for frame_data in frames:
            frame_bytes = FaceProcessingUtils.decode_base64_frame(
                frame_data["frame_data"]
            )
            timestamp = datetime.fromisoformat(
                frame_data.get("timestamp", datetime.utcnow().isoformat())
            )

            # Create single frame stream
            async def single_frame_stream():
                yield frame_bytes

            # Process frame
            async for alert in gateway_adapter.receive_frame_stream(
                exam_session_id=exam_session_id,
                candidate_id=candidate_id,
                frame_stream=single_frame_stream(),
                db_session=db
            ):
                if alert:
                    alerts.append(alert)

        await gateway_adapter.close()

        return {
            "status": "success",
            "frames_processed": len(frames),
            "alerts_generated": len(alerts)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch frame upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process frames: {str(e)}"
        )