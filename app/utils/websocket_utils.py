"""
WebSocket utilities for WebRTC frame streaming.
"""
import json
import asyncio
import logging
from typing import AsyncGenerator, Optional
from datetime import datetime
from fastapi import WebSocket

from app.utils.face_processing_utils import FaceProcessingUtils

logger = logging.getLogger(__name__)


class WebSocketFrameHandler:
    """Handler for WebSocket frame streaming operations."""

    @staticmethod
    async def send_connection_confirmation(
        websocket: WebSocket,
        exam_session_id: str,
        candidate_id: str
    ):
        """
        Send connection confirmation message.

        Args:
            websocket: WebSocket connection
            exam_session_id: Exam session ID
            candidate_id: Candidate ID
        """
        await websocket.send_json({
            "type": "connected",
            "exam_session_id": exam_session_id,
            "candidate_id": candidate_id,
            "timestamp": datetime.utcnow().isoformat()
        })

    @staticmethod
    async def send_error(websocket: WebSocket, message: str):
        """
        Send error message and close connection.

        Args:
            websocket: WebSocket connection
            message: Error message
        """
        await websocket.send_json({
            "type": "error",
            "message": message
        })
        await websocket.close()

    @staticmethod
    async def send_alert(
        websocket: WebSocket,
        alert
    ):
        """
        Send alert to WebSocket client.

        Args:
            websocket: WebSocket connection
            alert: AlertEvent object
        """
        await websocket.send_json({
            "type": "alert",
            "event_id": alert.event_id,
            "event_type": alert.event_type.value,
            "severity_score": alert.severity_score,
            "timestamp": alert.timestamp.isoformat(),
            "message": f"Alert: {alert.event_type.value}"
        })

    @staticmethod
    async def create_frame_stream_generator(
        websocket: WebSocket,
        timeout: float = 30.0
    ) -> AsyncGenerator[bytes, None]:
        """
        Create async generator that yields frames from WebSocket messages.

        Args:
            websocket: WebSocket connection
            timeout: Timeout in seconds for receiving messages

        Yields:
            Frame bytes
        """
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=timeout
                )

                data = json.loads(message)

                if data.get("type") == "frame":
                    # Decode base64 frame data
                    frame_bytes = FaceProcessingUtils.decode_base64_frame(
                        data["frame_data"]
                    )
                    yield frame_bytes

                elif data.get("type") == "ping":
                    # Heartbeat - respond with pong
                    await websocket.send_json({"type": "pong"})

                elif data.get("type") == "close":
                    # Client requested close
                    break

                else:
                    logger.warning(f"Unknown message type: {data.get('type')}")

            except asyncio.TimeoutError:
                # Send ping to check if connection is alive
                await websocket.send_json({"type": "ping"})
                continue

            except Exception as e:
                logger.error(f"Error in frame stream generator: {e}", exc_info=True)
                break

    @staticmethod
    def parse_frame_message(message: str) -> Optional[dict]:
        """
        Parse frame message from WebSocket.

        Args:
            message: JSON message string

        Returns:
            Parsed message dictionary or None if invalid
        """
        try:
            data = json.loads(message)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse frame message: {e}")
            return None

    @staticmethod
    def decode_frame_from_message(message_data: dict) -> Optional[bytes]:
        """
        Decode frame bytes from message data.

        Args:
            message_data: Parsed message dictionary

        Returns:
            Frame bytes or None if decoding fails
        """
        try:
            if message_data.get("type") != "frame":
                return None

            frame_data = message_data.get("frame_data")
            if not frame_data:
                return None

            return FaceProcessingUtils.decode_base64_frame(frame_data)

        except Exception as e:
            logger.error(f"Failed to decode frame: {e}")
            return None