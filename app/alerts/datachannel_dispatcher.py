"""
Alert Dispatcher for WebRTC DataChannels.
"""
import json
import logging
from typing import Optional
from app.schemas import AlertEvent

logger = logging.getLogger(__name__)

class DataChannelAlertDispatcher:
    """
    Dispatches alerts to the frontend via WebRTC DataChannels.
    """
    def __init__(self, webrtc_manager):
        self.webrtc_manager = webrtc_manager

    async def dispatch(self, alert: AlertEvent):
        """
        Send alert to the specific session's DataChannel.
        """
        if not alert:
            return

        try:
            # Format alert for frontend
            message = {
                "type": "alert",
                "event_type": alert.event_type.value,
                "severity_score": alert.severity_score,
                "message": f"Detected {alert.event_type.value}",
                "timestamp": alert.timestamp.isoformat(),
                "evidence": alert.evidence.model_dump(exclude_none=True)
            }
            
            await self.webrtc_manager.send_data(
                alert.exam_session_id, 
                json.dumps(message)
            )
            logger.debug(f"Dispatched alert {alert.event_type} to session {alert.exam_session_id}")
            
        except Exception as e:
            logger.error(f"Failed to dispatch alert via DataChannel: {e}")
