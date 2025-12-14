"""
Session management service for exam sessions.
"""
import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import redis.asyncio as aioredis

from app.utils.redis_client import get_redis
from app.config import settings

logger = logging.getLogger(__name__)


class SessionService:
    """Service for managing exam session state."""

    def __init__(self):
        """Initialize session service."""
        self.redis_prefix = "session:"
        self.session_ttl = 3600 * 4  # 4 hours default TTL

    async def start_session(
        self,
        exam_session_id: str,
        candidate_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize session state in Redis.

        Args:
            exam_session_id: Exam session identifier
            candidate_id: Candidate identifier
            config: Optional session configuration
        """
        redis = await get_redis()

        session_data = {
            "exam_session_id": exam_session_id,
            "candidate_id": candidate_id,
            "status": "active",
            "started_at": datetime.utcnow().isoformat(),
            "config": json.dumps(config or {}),  # Serialize dict to JSON string
            "frame_count": "0",  # Store as string
            "verification_count": "0",  # Store as string
            "last_verification": ""
        }

        key = f"{self.redis_prefix}{exam_session_id}"
        await redis.hset(key, mapping=session_data)
        await redis.expire(key, self.session_ttl)

        logger.info(f"Started session {exam_session_id} for candidate {candidate_id}")

    async def get_session(self, exam_session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session state.

        Args:
            exam_session_id: Exam session identifier

        Returns:
            Session data dictionary or None if not found
        """
        redis = await get_redis()

        key = f"{self.redis_prefix}{exam_session_id}"
        session_data = await redis.hgetall(key)

        if not session_data:
            return None

        # Decode bytes to strings and deserialize JSON fields
        decoded = {}
        for k, v in session_data.items():
            key_str = k.decode('utf-8') if isinstance(k, bytes) else k
            val_str = v.decode('utf-8') if isinstance(v, bytes) else v

            # Deserialize config JSON
            if key_str == 'config' and val_str:
                try:
                    decoded[key_str] = json.loads(val_str)
                except json.JSONDecodeError:
                    decoded[key_str] = {}
            # Convert numeric strings to integers
            elif key_str in ['frame_count', 'verification_count']:
                try:
                    decoded[key_str] = int(val_str)
                except (ValueError, TypeError):
                    decoded[key_str] = 0
            else:
                decoded[key_str] = val_str

        return decoded

    async def update_session(
        self,
        exam_session_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update session state.

        Args:
            exam_session_id: Exam session identifier
            updates: Dictionary of fields to update
        """
        redis = await get_redis()

        # Serialize any dict values to JSON
        serialized_updates = {}
        for k, v in updates.items():
            if isinstance(v, dict):
                serialized_updates[k] = json.dumps(v)
            elif isinstance(v, (int, float)):
                serialized_updates[k] = str(v)
            elif v is None:
                serialized_updates[k] = ""
            else:
                serialized_updates[k] = str(v)

        key = f"{self.redis_prefix}{exam_session_id}"
        await redis.hset(key, mapping=serialized_updates)

        # Refresh TTL
        await redis.expire(key, self.session_ttl)

    async def increment_frame_count(self, exam_session_id: str) -> int:
        """
        Increment frame processing counter.

        Args:
            exam_session_id: Exam session identifier

        Returns:
            New frame count
        """
        redis = await get_redis()

        key = f"{self.redis_prefix}{exam_session_id}"
        count = await redis.hincrby(key, "frame_count", 1)

        return int(count)

    async def increment_verification_count(
        self,
        exam_session_id: str,
        last_verification_time: Optional[datetime] = None
    ) -> int:
        """
        Increment verification counter.

        Args:
            exam_session_id: Exam session identifier
            last_verification_time: Timestamp of last verification

        Returns:
            New verification count
        """
        redis = await get_redis()

        key = f"{self.redis_prefix}{exam_session_id}"
        count = await redis.hincrby(key, "verification_count", 1)

        if last_verification_time:
            await redis.hset(key, "last_verification", last_verification_time.isoformat())

        return int(count)

    async def end_session(self, exam_session_id: str) -> None:
        """
        End session and clean up state.

        Args:
            exam_session_id: Exam session identifier
        """
        redis = await get_redis()

        key = f"{self.redis_prefix}{exam_session_id}"

        # Update status
        await redis.hset(key, mapping={
            "status": "ended",
            "ended_at": datetime.utcnow().isoformat()
        })

        # Set shorter TTL for ended sessions (1 hour)
        await redis.expire(key, 3600)

        logger.info(f"Ended session {exam_session_id}")

    async def is_session_active(self, exam_session_id: str) -> bool:
        """
        Check if session is active.

        Args:
            exam_session_id: Exam session identifier

        Returns:
            True if session exists and is active
        """
        session = await self.get_session(exam_session_id)
        if not session:
            return False

        status = session.get("status", "unknown")
        return status == "active"

    async def get_candidate_id(self, exam_session_id: str) -> Optional[str]:
        """
        Get candidate ID for a session.

        Args:
            exam_session_id: Exam session identifier

        Returns:
            Candidate ID or None
        """
        session = await self.get_session(exam_session_id)
        if not session:
            return None

        return session.get("candidate_id")