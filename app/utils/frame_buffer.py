"""
Frame buffer manager for ephemeral frame storage in Redis with JSON file backup.
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
import os
import asyncio
import aiofiles
from pathlib import Path

from app.config import settings
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)


class Frame:
    """Frame data structure."""

    def __init__(
            self,
            frame_id: str,
            exam_session_id: str,
            timestamp: datetime,
            metadata: Optional[Dict[str, Any]] = None
    ):
        self.frame_id = frame_id
        self.exam_session_id = exam_session_id
        self.timestamp = timestamp
        self.metadata = metadata or {}


class FrameBuffer:
    """Manages ephemeral frame storage in Redis with JSON file backup."""

    def __init__(self, redis_client: Optional[aioredis.Redis] = None, enable_file_storage: bool = True):
        """
        Initialize frame buffer.

        Args:
            redis_client: Optional Redis client (will create if not provided)
            enable_file_storage: Enable JSON file storage alongside Redis (default True)
        """
        self.redis_client = None
        self.retention_seconds = settings.frame_buffer_retention_seconds
        self._redis_pool = None
        self.enable_file_storage = enable_file_storage

        # Directory to dump session frames when clearing
        self.dump_dir = os.path.join(os.getcwd(), "analysis_dumps")
        os.makedirs(self.dump_dir, exist_ok=True)

        # Directory for active session file storage
        self.storage_dir = os.path.join(os.getcwd(), "frame_storage")
        os.makedirs(self.storage_dir, exist_ok=True)

        # Write buffer for batch file operations
        self._write_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._buffer_size = 50  # Flush to file after this many frames
        self._buffer_lock = asyncio.Lock()

    async def _get_redis(self):
        """Get or create Redis client."""
        if self.redis_client is None:
            self.redis_client = await get_redis()
        return self.redis_client

    def _get_session_file_path(self, exam_session_id: str) -> str:
        """Get the JSON file path for a session."""
        return os.path.join(self.storage_dir, f"{exam_session_id}.jsonl")

    async def _append_to_file(self, exam_session_id: str, data: Dict[str, Any]):
        """
        Append a frame to the session's JSONL file.
        Uses JSONL (JSON Lines) format for efficient appending.
        """
        if not self.enable_file_storage:
            return

        filepath = self._get_session_file_path(exam_session_id)

        try:
            async with aiofiles.open(filepath, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to append frame to file {filepath}: {e}", exc_info=True)

    async def _flush_write_buffer(self, exam_session_id: str):
        """Flush the write buffer for a session to file."""
        async with self._buffer_lock:
            if exam_session_id not in self._write_buffer:
                return

            frames = self._write_buffer[exam_session_id]
            if not frames:
                return

            filepath = self._get_session_file_path(exam_session_id)

            try:
                async with aiofiles.open(filepath, 'a', encoding='utf-8') as f:
                    for frame in frames:
                        await f.write(json.dumps(frame, ensure_ascii=False) + '\n')

                logger.debug(f"Flushed {len(frames)} frames to {filepath}")
                self._write_buffer[exam_session_id] = []
            except Exception as e:
                logger.error(f"Failed to flush buffer to {filepath}: {e}", exc_info=True)

    async def _buffer_frame_for_file(self, exam_session_id: str, data: Dict[str, Any]):
        """
        Buffer frame for batch writing to file.
        Flushes when buffer reaches threshold.
        """
        if not self.enable_file_storage:
            return

        async with self._buffer_lock:
            if exam_session_id not in self._write_buffer:
                self._write_buffer[exam_session_id] = []

            self._write_buffer[exam_session_id].append(data)

            # Flush if buffer is full
            if len(self._write_buffer[exam_session_id]) >= self._buffer_size:
                await self._flush_write_buffer(exam_session_id)

    async def store_metadata_frame(
            self,
            exam_session_id: str,
            data: Dict[str, Any],
            timestamp: Optional[datetime] = None,
            storage_mode: str = "file"  # "redis", "file", or "both"
    ) -> str:
        """
        Store metadata-only frame (extracted features) in Redis and/or file.

        Args:
            exam_session_id: Exam session identifier
            data: Dictionary of extracted features (face_present, no_of_face, face_x, etc.)
            timestamp: Optional timestamp (will use current time if not provided)
            storage_mode: Where to store ("redis", "file", or "both")

        Returns:
            Frame ID (timestamp in milliseconds)
        """
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Use timestamp in milliseconds as frame_id for uniqueness and ordering
        frame_id = str(int(timestamp.timestamp() * 1000))

        # Ensure session_id is in the data
        if "session_id" not in data:
            data["session_id"] = exam_session_id

        # Add timestamp and frame_id to the data
        data["timestamp"] = timestamp.isoformat()
        data["frame_id"] = frame_id

        # Store in Redis
        if storage_mode in ["redis", "both"]:
            try:
                redis = await self._get_redis()

                # Store the metadata frame
                frame_key = f"frame_meta:{exam_session_id}:{frame_id}"
                await redis.set(
                    frame_key,
                    json.dumps(data, ensure_ascii=False),
                    ex=self.retention_seconds
                )

                # Add to sorted set for time-range queries (score is timestamp in seconds)
                sorted_set_key = f"frames_meta:{exam_session_id}"
                await redis.zadd(
                    sorted_set_key,
                    {frame_id: timestamp.timestamp()}
                )

                # Set TTL on sorted set as well
                await redis.expire(sorted_set_key, self.retention_seconds)

                logger.debug(f"Stored metadata frame {frame_id} in Redis for session {exam_session_id}")
            except Exception as e:
                logger.error(f"Failed to store metadata frame in Redis: {e}", exc_info=True)
                # If Redis fails and we're in "redis" mode, raise
                if storage_mode == "redis":
                    raise

        # Store in file
        if storage_mode in ["file", "both"] and self.enable_file_storage:
            try:
                await self._buffer_frame_for_file(exam_session_id, data)
                logger.debug(f"Buffered metadata frame {frame_id} for file storage")
            except Exception as e:
                logger.error(f"Failed to buffer frame for file storage: {e}", exc_info=True)
                # If file storage fails and we're in "file" mode, raise
                if storage_mode == "file":
                    raise

        return frame_id

    async def get_metadata_frames(
            self,
            exam_session_id: str,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            limit: int = 10000,
            source: str = "file"  # "redis", "file", or "auto"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve metadata-only frames (JSON messages) in time range.
        Returns list of dicts (no image decoding).

        Args:
            exam_session_id: Exam session identifier
            start_time: Optional start time for filtering frames
            end_time: Optional end time for filtering frames
            limit: Maximum number of frames to retrieve (default 10000)
            source: Where to retrieve from ("redis", "file", or "auto" - tries Redis first)

        Returns:
            List of frame metadata dictionaries, sorted by timestamp (oldest first)
        """
        frames = []

        # Auto mode: try Redis first, fall back to file
        if source in ["auto", "redis"]:
            frames = await self._get_frames_from_redis(
                exam_session_id, start_time, end_time, limit
            )

            if frames or source == "redis":
                return frames

        # Try file if Redis returned nothing or source is "file"
        if source in ["auto", "file"]:
            frames = await self._get_frames_from_file(
                exam_session_id, start_time, end_time, limit
            )

        return frames

    async def _get_frames_from_redis(
            self,
            exam_session_id: str,
            start_time: Optional[datetime],
            end_time: Optional[datetime],
            limit: int
    ) -> List[Dict[str, Any]]:
        """Retrieve frames from Redis."""
        try:
            redis = await self._get_redis()
            sorted_set_key = f"frames_meta:{exam_session_id}"

            # Convert datetime to timestamp scores for Redis ZRANGEBYSCORE
            min_score = start_time.timestamp() if start_time else "-inf"
            max_score = end_time.timestamp() if end_time else "+inf"

            # Get frame IDs within the time range, ordered by timestamp
            frame_ids = await redis.zrangebyscore(
                sorted_set_key,
                min=min_score,
                max=max_score,
                start=0,
                num=limit
            )

            if not frame_ids:
                logger.debug(f"No metadata frames found in Redis for session {exam_session_id}")
                return []

            # Retrieve all frame data
            frames = []
            for frame_id in frame_ids:
                frame_id_str = frame_id.decode('utf-8') if isinstance(frame_id, bytes) else frame_id
                frame_key = f"frame_meta:{exam_session_id}:{frame_id_str}"

                frame_data = await redis.get(frame_key)
                if frame_data:
                    try:
                        frame_dict = json.loads(frame_data)
                        frames.append(frame_dict)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode frame {frame_id_str}: {e}")
                        continue

            logger.debug(f"Retrieved {len(frames)} metadata frames from Redis for session {exam_session_id}")
            return frames

        except Exception as e:
            logger.error(f"Failed to retrieve metadata frames from Redis: {e}", exc_info=True)
            return []

    async def _get_frames_from_file(
            self,
            exam_session_id: str,
            start_time: Optional[datetime],
            end_time: Optional[datetime],
            limit: int
    ) -> List[Dict[str, Any]]:
        """Retrieve frames from JSONL file."""
        if not self.enable_file_storage:
            return []

        # First, flush any buffered frames for this session
        await self._flush_write_buffer(exam_session_id)

        filepath = self._get_session_file_path(exam_session_id)

        if not os.path.exists(filepath):
            logger.debug(f"No file storage found for session {exam_session_id}")
            return []

        try:
            frames = []
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        frame = json.loads(line)

                        # Apply time filtering if specified
                        if start_time or end_time:
                            frame_time = datetime.fromisoformat(frame.get('timestamp', ''))
                            if start_time and frame_time < start_time:
                                continue
                            if end_time and frame_time > end_time:
                                continue

                        frames.append(frame)

                        # Apply limit
                        if len(frames) >= limit:
                            break

                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Failed to parse line in {filepath}: {e}")
                        continue

            logger.debug(f"Retrieved {len(frames)} metadata frames from file for session {exam_session_id}")
            return frames

        except Exception as e:
            logger.error(f"Failed to retrieve frames from file {filepath}: {e}", exc_info=True)
            return []

    async def get_frame_count(self, exam_session_id: str, source: str = "auto") -> int:
        """
        Get the total number of frames stored for a session.

        Args:
            exam_session_id: Exam session identifier
            source: Where to count from ("redis", "file", or "auto")

        Returns:
            Total frame count
        """
        if source in ["auto", "redis"]:
            try:
                redis = await self._get_redis()
                sorted_set_key = f"frames_meta:{exam_session_id}"
                count = await redis.zcard(sorted_set_key)
                if count > 0 or source == "redis":
                    return count
            except Exception as e:
                logger.error(f"Failed to get frame count from Redis: {e}")

        # Try file if Redis returned 0 or source is "file"
        if source in ["auto", "file"] and self.enable_file_storage:
            await self._flush_write_buffer(exam_session_id)
            filepath = self._get_session_file_path(exam_session_id)

            if os.path.exists(filepath):
                try:
                    count = 0
                    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                        async for line in f:
                            if line.strip():
                                count += 1
                    return count
                except Exception as e:
                    logger.error(f"Failed to count frames in file: {e}")

        return 0

    async def clear_session_frames(self, exam_session_id: str, clear_file: bool = False) -> int:
        """
        Clear all frames for a session from Redis and optionally from file.

        Args:
            exam_session_id: Exam session identifier
            clear_file: Whether to also delete the file storage (default False)

        Returns:
            Number of frames deleted
        """
        redis = await self._get_redis()
        deleted = 0

        try:
            # Clear Redis frames
            sorted_set_key = f"frames:{exam_session_id}"
            frame_ids = await redis.zrange(sorted_set_key, 0, -1)

            # Delete individual frame keys
            for frame_id in frame_ids:
                frame_id_str = frame_id.decode('utf-8') if isinstance(frame_id, bytes) else frame_id
                key = f"frame:{exam_session_id}:{frame_id_str}"
                if await redis.delete(key):
                    deleted += 1

            # Delete sorted set
            await redis.delete(sorted_set_key)

            # Also clear metadata frames
            meta_sorted = f"frames_meta:{exam_session_id}"
            meta_ids = await redis.zrange(meta_sorted, 0, -1)
            for mid in meta_ids:
                mid_str = mid.decode('utf-8') if isinstance(mid, bytes) else mid
                await redis.delete(f"frame_meta:{exam_session_id}:{mid_str}")
            await redis.delete(meta_sorted)

            logger.info(f"Cleared {deleted} frames from Redis for session {exam_session_id}")

        except Exception as e:
            logger.error(f"Failed to clear Redis frames: {e}", exc_info=True)

        # Clear write buffer
        async with self._buffer_lock:
            if exam_session_id in self._write_buffer:
                del self._write_buffer[exam_session_id]

        # Optionally clear file storage
        if clear_file and self.enable_file_storage:
            filepath = self._get_session_file_path(exam_session_id)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.info(f"Deleted file storage for session {exam_session_id}")
                except Exception as e:
                    logger.error(f"Failed to delete file {filepath}: {e}")

        return deleted

    async def dump_metadata_to_file(
            self,
            exam_session_id: str,
            filepath: Optional[str] = None,
            source: str = "auto"
    ) -> str:
        """
        Dump all metadata frames for a session to a JSON file and return path.

        Args:
            exam_session_id: Exam session identifier
            filepath: Optional custom filepath (defaults to dump_dir)
            source: Where to retrieve from ("redis", "file", or "auto")

        Returns:
            Path to the dumped file, or empty string if failed
        """
        frames = await self.get_metadata_frames(exam_session_id, limit=100000, source=source)

        if not frames:
            return ""

        filepath = filepath or os.path.join(self.dump_dir, f"{exam_session_id}_frames.json")

        try:
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(frames, ensure_ascii=False, indent=2))

            logger.info(f"Dumped {len(frames)} metadata frames for {exam_session_id} to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to dump frames to file: {e}", exc_info=True)
            return ""

    async def convert_jsonl_to_json(self, exam_session_id: str, output_path: Optional[str] = None) -> str:
        """
        Convert the JSONL storage file to a standard JSON array file.

        Args:
            exam_session_id: Exam session identifier
            output_path: Optional output path (defaults to dump_dir)

        Returns:
            Path to the converted file
        """
        frames = await self._get_frames_from_file(exam_session_id, None, None, 1000000)

        if not frames:
            return ""

        output_path = output_path or os.path.join(self.dump_dir, f"{exam_session_id}_converted.json")

        try:
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(frames, ensure_ascii=False, indent=2))

            logger.info(f"Converted JSONL to JSON for {exam_session_id}: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to convert JSONL to JSON: {e}", exc_info=True)
            return ""

    async def get_session_statistics(self, exam_session_id: str) -> Dict[str, Any]:
        """
        Get statistics about stored frames for a session.

        Returns:
            Dictionary with frame counts, time range, and storage info
        """
        stats = {
            "session_id": exam_session_id,
            "redis_count": 0,
            "file_count": 0,
            "total_count": 0,
            "redis_available": False,
            "file_available": False,
            "time_range": None
        }

        # Check Redis
        try:
            redis_count = await self.get_frame_count(exam_session_id, source="redis")
            stats["redis_count"] = redis_count
            stats["redis_available"] = redis_count > 0
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")

        # Check file
        if self.enable_file_storage:
            try:
                file_count = await self.get_frame_count(exam_session_id, source="file")
                stats["file_count"] = file_count
                stats["file_available"] = file_count > 0
            except Exception as e:
                logger.error(f"Failed to get file stats: {e}")

        # Get time range from available source
        frames = await self.get_metadata_frames(exam_session_id, limit=1, source="auto")
        if frames:
            last_frames = await self.get_metadata_frames(
                exam_session_id,
                limit=1,
                source="auto"
            )
            if last_frames:
                stats["time_range"] = {
                    "start": frames[0].get("timestamp"),
                    "end": last_frames[-1].get("timestamp")
                }

        stats["total_count"] = max(stats["redis_count"], stats["file_count"])

        return stats

    async def close(self):
        """Close Redis connection and flush any pending writes."""
        # Flush all pending writes
        for session_id in list(self._write_buffer.keys()):
            await self._flush_write_buffer(session_id)

        if self._redis_pool:
            await self._redis_pool.close()
            self._redis_pool = None