"""
Session management service for exam sessions with dual storage support.
"""
import logging
import json
import os
from typing import Optional, Dict, Any, Literal
from datetime import datetime, timedelta
import redis.asyncio as aioredis
import aiofiles
from pathlib import Path
import asyncio

from app.utils.redis_client import get_redis
from app.config import settings

logger = logging.getLogger(__name__)


class SessionService:
    """Service for managing exam session state with Redis and file storage."""

    def __init__(self, storage_mode: Literal["redis", "file", "both"] = "file"):
        """
        Initialize session service.

        Args:
            storage_mode: Storage backend to use
                - "redis": Use only Redis (ephemeral)
                - "file": Use only JSON files (persistent, default)
                - "both": Use both Redis and files (redundant)
        """
        self.storage_mode = storage_mode
        self.redis_prefix = "session:"
        self.session_ttl = 3600 * 4  # 4 hours default TTL

        # File storage setup
        self.sessions_dir = os.path.join(os.getcwd(), "session_storage")
        os.makedirs(self.sessions_dir, exist_ok=True)

        # Lock for file operations to prevent race conditions
        self._file_locks: Dict[str, asyncio.Lock] = {}

    def _get_session_file_path(self, exam_session_id: str) -> str:
        """Get the JSON file path for a session."""
        return os.path.join(self.sessions_dir, f"{exam_session_id}.json")

    async def _get_file_lock(self, exam_session_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific session file."""
        if exam_session_id not in self._file_locks:
            self._file_locks[exam_session_id] = asyncio.Lock()
        return self._file_locks[exam_session_id]

    async def _read_session_file(self, exam_session_id: str) -> Optional[Dict[str, Any]]:
        """Read session data from file."""
        filepath = self._get_session_file_path(exam_session_id)

        if not os.path.exists(filepath):
            return None

        try:
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to read session file {filepath}: {e}", exc_info=True)
            return None

    async def _write_session_file(self, exam_session_id: str, data: Dict[str, Any]) -> None:
        """Write session data to file."""
        filepath = self._get_session_file_path(exam_session_id)

        try:
            # Add last_updated timestamp
            data["last_updated"] = datetime.utcnow().isoformat()

            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"Failed to write session file {filepath}: {e}", exc_info=True)
            raise

    async def _delete_session_file(self, exam_session_id: str) -> None:
        """Delete session file."""
        filepath = self._get_session_file_path(exam_session_id)

        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Deleted session file: {filepath}")
            except Exception as e:
                logger.error(f"Failed to delete session file {filepath}: {e}")

    async def start_session(
            self,
            exam_session_id: str,
            candidate_id: str,
            config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize session state in Redis and/or file.

        Args:
            exam_session_id: Exam session identifier
            candidate_id: Candidate identifier
            config: Optional session configuration
        """
        session_data = {
            "exam_session_id": exam_session_id,
            "candidate_id": candidate_id,
            "status": "active",
            "started_at": datetime.utcnow().isoformat(),
            "config": config or {},
            "frame_count": 0,
            "verification_count": 0,
            "last_verification": None
        }

        # Store in Redis
        if self.storage_mode in ["redis", "both"]:
            try:
                redis = await get_redis()

                # Redis requires string values
                redis_data = {
                    "exam_session_id": exam_session_id,
                    "candidate_id": candidate_id,
                    "status": "active",
                    "started_at": session_data["started_at"],
                    "config": json.dumps(config or {}),
                    "frame_count": "0",
                    "verification_count": "0",
                    "last_verification": ""
                }

                key = f"{self.redis_prefix}{exam_session_id}"
                await redis.hset(key, mapping=redis_data)
                await redis.expire(key, self.session_ttl)

                logger.debug(f"Started session {exam_session_id} in Redis")
            except Exception as e:
                logger.error(f"Failed to start session in Redis: {e}", exc_info=True)
                if self.storage_mode == "redis":
                    raise

        # Store in file
        if self.storage_mode in ["file", "both"]:
            try:
                lock = await self._get_file_lock(exam_session_id)
                async with lock:
                    await self._write_session_file(exam_session_id, session_data)
                logger.debug(f"Started session {exam_session_id} in file")
            except Exception as e:
                logger.error(f"Failed to start session in file: {e}", exc_info=True)
                if self.storage_mode == "file":
                    raise

        logger.info(f"Started session {exam_session_id} for candidate {candidate_id} (mode: {self.storage_mode})")

    async def get_session(self, exam_session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session state from Redis or file (auto fallback).

        Args:
            exam_session_id: Exam session identifier

        Returns:
            Session data dictionary or None if not found
        """
        # Try Redis first if enabled
        if self.storage_mode in ["redis", "both"]:
            try:
                redis = await get_redis()
                key = f"{self.redis_prefix}{exam_session_id}"
                session_data = await redis.hgetall(key)

                if session_data:
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
                        # Handle None/empty string for last_verification
                        elif key_str == 'last_verification' and not val_str:
                            decoded[key_str] = None
                        else:
                            decoded[key_str] = val_str

                    logger.debug(f"Retrieved session {exam_session_id} from Redis")
                    return decoded
            except Exception as e:
                logger.error(f"Failed to get session from Redis: {e}", exc_info=True)

        # Fall back to file or use file directly
        if self.storage_mode in ["file", "both"]:
            try:
                lock = await self._get_file_lock(exam_session_id)
                async with lock:
                    session_data = await self._read_session_file(exam_session_id)
                    if session_data:
                        logger.debug(f"Retrieved session {exam_session_id} from file")
                        return session_data
            except Exception as e:
                logger.error(f"Failed to get session from file: {e}", exc_info=True)

        return None

    async def update_session(
            self,
            exam_session_id: str,
            updates: Dict[str, Any]
    ) -> None:
        """
        Update session state in Redis and/or file.

        Args:
            exam_session_id: Exam session identifier
            updates: Dictionary of fields to update
        """
        # Update in Redis
        if self.storage_mode in ["redis", "both"]:
            try:
                redis = await get_redis()

                # Serialize any dict values to JSON for Redis
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

                logger.debug(f"Updated session {exam_session_id} in Redis")
            except Exception as e:
                logger.error(f"Failed to update session in Redis: {e}", exc_info=True)
                if self.storage_mode == "redis":
                    raise

        # Update in file
        if self.storage_mode in ["file", "both"]:
            try:
                lock = await self._get_file_lock(exam_session_id)
                async with lock:
                    # Read existing data
                    session_data = await self._read_session_file(exam_session_id)

                    if session_data is None:
                        logger.warning(f"Session {exam_session_id} not found in file, cannot update")
                        return

                    # Apply updates
                    session_data.update(updates)

                    # Write back
                    await self._write_session_file(exam_session_id, session_data)

                logger.debug(f"Updated session {exam_session_id} in file")
            except Exception as e:
                logger.error(f"Failed to update session in file: {e}", exc_info=True)
                if self.storage_mode == "file":
                    raise

    async def increment_frame_count(self, exam_session_id: str) -> int:
        """
        Increment frame processing counter.

        Args:
            exam_session_id: Exam session identifier

        Returns:
            New frame count
        """
        # Increment in Redis
        if self.storage_mode in ["redis", "both"]:
            try:
                redis = await get_redis()
                key = f"{self.redis_prefix}{exam_session_id}"
                count = await redis.hincrby(key, "frame_count", 1)

                # If using file as well, sync the count
                if self.storage_mode == "both":
                    await self.update_session(exam_session_id, {"frame_count": int(count)})

                return int(count)
            except Exception as e:
                logger.error(f"Failed to increment frame count in Redis: {e}", exc_info=True)
                if self.storage_mode == "redis":
                    raise

        # Increment in file
        if self.storage_mode == "file":
            try:
                lock = await self._get_file_lock(exam_session_id)
                async with lock:
                    session_data = await self._read_session_file(exam_session_id)

                    if session_data is None:
                        logger.warning(f"Session {exam_session_id} not found, cannot increment frame count")
                        return 0

                    current_count = session_data.get("frame_count", 0)
                    new_count = current_count + 1
                    session_data["frame_count"] = new_count

                    await self._write_session_file(exam_session_id, session_data)

                    return new_count
            except Exception as e:
                logger.error(f"Failed to increment frame count in file: {e}", exc_info=True)
                raise

        return 0

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
        verification_iso = last_verification_time.isoformat() if last_verification_time else None

        # Increment in Redis
        if self.storage_mode in ["redis", "both"]:
            try:
                redis = await get_redis()
                key = f"{self.redis_prefix}{exam_session_id}"
                count = await redis.hincrby(key, "verification_count", 1)

                if last_verification_time:
                    await redis.hset(key, "last_verification", verification_iso)

                # If using file as well, sync
                if self.storage_mode == "both":
                    updates = {"verification_count": int(count)}
                    if verification_iso:
                        updates["last_verification"] = verification_iso
                    await self.update_session(exam_session_id, updates)

                return int(count)
            except Exception as e:
                logger.error(f"Failed to increment verification count in Redis: {e}", exc_info=True)
                if self.storage_mode == "redis":
                    raise

        # Increment in file
        if self.storage_mode == "file":
            try:
                lock = await self._get_file_lock(exam_session_id)
                async with lock:
                    session_data = await self._read_session_file(exam_session_id)

                    if session_data is None:
                        logger.warning(f"Session {exam_session_id} not found, cannot increment verification count")
                        return 0

                    current_count = session_data.get("verification_count", 0)
                    new_count = current_count + 1
                    session_data["verification_count"] = new_count

                    if verification_iso:
                        session_data["last_verification"] = verification_iso

                    await self._write_session_file(exam_session_id, session_data)

                    return new_count
            except Exception as e:
                logger.error(f"Failed to increment verification count in file: {e}", exc_info=True)
                raise

        return 0

    async def end_session(self, exam_session_id: str) -> None:
        """
        End session and clean up state.

        Args:
            exam_session_id: Exam session identifier
        """
        ended_at = datetime.utcnow().isoformat()

        # Update in Redis
        if self.storage_mode in ["redis", "both"]:
            try:
                redis = await get_redis()
                key = f"{self.redis_prefix}{exam_session_id}"

                # Update status
                await redis.hset(key, mapping={
                    "status": "ended",
                    "ended_at": ended_at
                })

                # Set shorter TTL for ended sessions (1 hour)
                await redis.expire(key, 3600)

                logger.debug(f"Ended session {exam_session_id} in Redis")
            except Exception as e:
                logger.error(f"Failed to end session in Redis: {e}", exc_info=True)
                if self.storage_mode == "redis":
                    raise

        # Update in file
        if self.storage_mode in ["file", "both"]:
            try:
                lock = await self._get_file_lock(exam_session_id)
                async with lock:
                    session_data = await self._read_session_file(exam_session_id)

                    if session_data:
                        session_data["status"] = "ended"
                        session_data["ended_at"] = ended_at
                        await self._write_session_file(exam_session_id, session_data)

                logger.debug(f"Ended session {exam_session_id} in file")
            except Exception as e:
                logger.error(f"Failed to end session in file: {e}", exc_info=True)
                if self.storage_mode == "file":
                    raise

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

    async def list_active_sessions(self) -> list[Dict[str, Any]]:
        """
        List all active sessions.

        Returns:
            List of active session data
        """
        active_sessions = []

        if self.storage_mode in ["file", "both"]:
            try:
                # Scan session files
                for filename in os.listdir(self.sessions_dir):
                    if filename.endswith('.json'):
                        session_id = filename[:-5]  # Remove .json extension
                        session = await self.get_session(session_id)
                        if session and session.get("status") == "active":
                            active_sessions.append(session)
            except Exception as e:
                logger.error(f"Failed to list sessions from files: {e}", exc_info=True)

        elif self.storage_mode == "redis":
            try:
                redis = await get_redis()
                # Scan for session keys
                pattern = f"{self.redis_prefix}*"
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(cursor, match=pattern, count=100)
                    for key in keys:
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                        session_id = key_str.replace(self.redis_prefix, "")
                        session = await self.get_session(session_id)
                        if session and session.get("status") == "active":
                            active_sessions.append(session)

                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Failed to list sessions from Redis: {e}", exc_info=True)

        return active_sessions

    async def delete_session(self, exam_session_id: str, delete_file: bool = True) -> None:
        """
        Completely delete a session from storage.

        Args:
            exam_session_id: Exam session identifier
            delete_file: Whether to delete the file storage (default True)
        """
        # Delete from Redis
        if self.storage_mode in ["redis", "both"]:
            try:
                redis = await get_redis()
                key = f"{self.redis_prefix}{exam_session_id}"
                await redis.delete(key)
                logger.info(f"Deleted session {exam_session_id} from Redis")
            except Exception as e:
                logger.error(f"Failed to delete session from Redis: {e}", exc_info=True)

        # Delete from file
        if delete_file and self.storage_mode in ["file", "both"]:
            try:
                await self._delete_session_file(exam_session_id)
            except Exception as e:
                logger.error(f"Failed to delete session file: {e}", exc_info=True)

        # Clean up lock
        if exam_session_id in self._file_locks:
            del self._file_locks[exam_session_id]

    async def get_session_statistics(self, exam_session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive statistics for a session.

        Args:
            exam_session_id: Exam session identifier

        Returns:
            Dictionary with session statistics or None if not found
        """
        session = await self.get_session(exam_session_id)
        if not session:
            return None

        stats = {
            "session_id": exam_session_id,
            "candidate_id": session.get("candidate_id"),
            "status": session.get("status"),
            "started_at": session.get("started_at"),
            "ended_at": session.get("ended_at"),
            "frame_count": session.get("frame_count", 0),
            "verification_count": session.get("verification_count", 0),
            "last_verification": session.get("last_verification"),
            "storage_mode": self.storage_mode
        }

        # Calculate duration if session has started
        if session.get("started_at"):
            try:
                start_time = datetime.fromisoformat(session["started_at"])
                if session.get("ended_at"):
                    end_time = datetime.fromisoformat(session["ended_at"])
                else:
                    end_time = datetime.utcnow()

                duration = (end_time - start_time).total_seconds()
                stats["duration_seconds"] = duration
            except Exception as e:
                logger.error(f"Failed to calculate duration: {e}")
                stats["duration_seconds"] = None

        return stats