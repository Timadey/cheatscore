"""
Frame buffer manager for ephemeral frame storage in Redis.
"""
import numpy as np
import json
import base64
import cv2
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import redis.asyncio as aioredis
import pickle

from app.config import settings

logger = logging.getLogger(__name__)


class Frame:
    """Frame data structure."""
    
    def __init__(
        self,
        frame_id: str,
        exam_session_id: str,
        frame_data: np.ndarray,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.frame_id = frame_id
        self.exam_session_id = exam_session_id
        self.frame_data = frame_data
        self.timestamp = timestamp
        self.metadata = metadata or {}


class FrameBuffer:
    """Manages ephemeral frame storage in Redis."""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        """
        Initialize frame buffer.
        
        Args:
            redis_client: Optional Redis client (will create if not provided)
        """
        self.redis_client = redis_client
        self.retention_seconds = settings.frame_buffer_retention_seconds
        self._redis_pool = None
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get or create Redis client."""
        if self.redis_client is None:
            if self._redis_pool is None:
                self._redis_pool = aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=False
                )
            return self._redis_pool
        return self.redis_client
    
    async def store_frame(
        self,
        exam_session_id: str,
        frame: np.ndarray,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store frame in Redis buffer.
        
        Args:
            exam_session_id: Exam session identifier
            frame: Frame image as numpy array
            timestamp: Frame timestamp (default: now)
            metadata: Optional frame metadata
            
        Returns:
            Frame ID
        """
        redis = await self._get_redis()
        timestamp = timestamp or datetime.utcnow()
        frame_id = f"FRAME-{exam_session_id}-{timestamp.timestamp()}"
        
        try:
            # Serialize frame (compress JPEG for efficiency)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Create frame data
            frame_data = {
                "frame_id": frame_id,
                "exam_session_id": exam_session_id,
                "timestamp": timestamp.isoformat(),
                "frame_bytes": base64.b64encode(frame_bytes).decode('utf-8'),
                "shape": list(frame.shape),
                "metadata": metadata or {}
            }
            
            # Store in Redis with TTL
            key = f"frame:{exam_session_id}:{frame_id}"
            await redis.setex(
                key,
                self.retention_seconds,
                json.dumps(frame_data)
            )
            
            # Add to sorted set for time-based retrieval
            sorted_set_key = f"frames:{exam_session_id}"
            await redis.zadd(
                sorted_set_key,
                {frame_id: timestamp.timestamp()}
            )
            await redis.expire(sorted_set_key, self.retention_seconds)
            
            logger.debug(f"Stored frame {frame_id} for session {exam_session_id}")
            return frame_id
            
        except Exception as e:
            logger.error(f"Failed to store frame: {e}", exc_info=True)
            raise
    
    async def get_frame(self, exam_session_id: str, frame_id: str) -> Optional[Frame]:
        """
        Retrieve a specific frame.
        
        Args:
            exam_session_id: Exam session identifier
            frame_id: Frame identifier
            
        Returns:
            Frame object or None if not found
        """
        redis = await self._get_redis()
        
        try:
            key = f"frame:{exam_session_id}:{frame_id}"
            data = await redis.get(key)
            
            if data is None:
                return None
            
            frame_data = json.loads(data)
            
            # Decode frame
            frame_bytes = base64.b64decode(frame_data["frame_bytes"])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return Frame(
                frame_id=frame_data["frame_id"],
                exam_session_id=frame_data["exam_session_id"],
                frame_data=frame,
                timestamp=datetime.fromisoformat(frame_data["timestamp"]),
                metadata=frame_data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to get frame {frame_id}: {e}", exc_info=True)
            return None
    
    async def get_frames(
        self,
        exam_session_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Frame]:
        """
        Retrieve frames in time range.
        
        Args:
            exam_session_id: Exam session identifier
            start_time: Start time (default: retention window ago)
            end_time: End time (default: now)
            limit: Maximum number of frames to return
            
        Returns:
            List of Frame objects
        """
        redis = await self._get_redis()
        
        try:
            end_time = end_time or datetime.utcnow()
            start_time = start_time or (end_time - timedelta(seconds=self.retention_seconds))
            
            sorted_set_key = f"frames:{exam_session_id}"
            
            # Get frame IDs in time range
            frame_ids = await redis.zrangebyscore(
                sorted_set_key,
                min=start_time.timestamp(),
                max=end_time.timestamp(),
                start=0,
                num=limit
            )
            
            # Decode frame IDs
            frame_ids = [fid.decode('utf-8') if isinstance(fid, bytes) else fid for fid in frame_ids]
            
            # Retrieve frames
            frames = []
            for frame_id in frame_ids:
                frame = await self.get_frame(exam_session_id, frame_id)
                if frame:
                    frames.append(frame)
            
            return frames
            
        except Exception as e:
            logger.error(f"Failed to get frames: {e}", exc_info=True)
            return []
    
    async def get_recent_frames(
        self,
        exam_session_id: str,
        window_seconds: Optional[int] = None
    ) -> List[Frame]:
        """
        Get recent frames within time window.
        
        Args:
            exam_session_id: Exam session identifier
            window_seconds: Time window in seconds (default: retention_seconds)
            
        Returns:
            List of recent Frame objects
        """
        window = window_seconds or self.retention_seconds
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=window)
        
        return await self.get_frames(exam_session_id, start_time, end_time)
    
    async def clear_session_frames(self, exam_session_id: str) -> int:
        """
        Clear all frames for a session.
        
        Args:
            exam_session_id: Exam session identifier
            
        Returns:
            Number of frames deleted
        """
        redis = await self._get_redis()
        
        try:
            sorted_set_key = f"frames:{exam_session_id}"
            frame_ids = await redis.zrange(sorted_set_key, 0, -1)
            
            # Delete individual frame keys
            deleted = 0
            for frame_id in frame_ids:
                frame_id_str = frame_id.decode('utf-8') if isinstance(frame_id, bytes) else frame_id
                key = f"frame:{exam_session_id}:{frame_id_str}"
                if await redis.delete(key):
                    deleted += 1
            
            # Delete sorted set
            await redis.delete(sorted_set_key)
            
            logger.info(f"Cleared {deleted} frames for session {exam_session_id}")
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to clear frames: {e}", exc_info=True)
            return 0
    
    async def close(self):
        """Close Redis connection."""
        if self._redis_pool:
            await self._redis_pool.close()
            self._redis_pool = None

