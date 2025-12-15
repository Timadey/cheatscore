"""
WebRTC Gateway Adapter for receiving and processing video/audio frames.
"""
import numpy as np
import cv2
import logging
from typing import AsyncIterator, Optional, Dict, Any
from datetime import datetime
import asyncio
import av
from io import BytesIO

from app.utils.frame_buffer import FrameBuffer
from app.services.session_service import SessionService
from app.inference.continuous_verifier import ContinuousVerifier
from app.services.verification_service import VerificationService
from app.utils.vector_db import VectorDB
from app.schemas import AlertEvent
from app.config import settings
from app.models import FaceEnrollment
from sqlalchemy import select

logger = logging.getLogger(__name__)


class WebRTCGatewayAdapter:
    """Adapter for receiving frames from WebRTC gateway."""
    
    def __init__(self, db_session=None):
        """
        Initialize gateway adapter.
        
        Args:
            db_session: Optional database session for loading embeddings
        """
        self.frame_buffer = FrameBuffer()
        self.session_service = SessionService()
        self.continuous_verifier = ContinuousVerifier()
        self.verification_service = VerificationService()
        self.db_session = db_session
        self._enrolled_embeddings_cache: Dict[str, np.ndarray] = {}
    
    async def receive_frame_stream(
        self,
        exam_session_id: str,
        candidate_id: str,
        frame_stream: AsyncIterator[bytes],
        frame_metadata: Optional[Dict[str, Any]] = None,
        db_session = None
    ) -> AsyncIterator[Optional[AlertEvent]]:
        """
        Receive and process frames from WebRTC stream.
        Decoupled architecture: Producer (WebRTC) -> Latest Frame -> Consumer (Verification)
        """
        # Ensure session is active
        if not await self.session_service.is_session_active(exam_session_id):
            logger.warning(f"Session {exam_session_id} is not active")
            return
        
        # Use provided db_session or instance db_session
        db = db_session or self.db_session
        if db is None:
            raise ValueError("Database session is required for loading enrolled embeddings")
        
        # Load enrolled embedding for verification
        enrolled_embedding = await self._load_enrolled_embedding(candidate_id, db)
        
        if enrolled_embedding is None:
            logger.error(f"No enrolled embedding found for candidate {candidate_id}")
            yield None
            return
        
        logger.info(f"Loaded enrolled embedding for candidate {candidate_id}, session {exam_session_id}")
        
        # Shared state
        latest_frame = {"data": None, "timestamp": None}
        processing_active = True
        
        # Producer Task: Read frames from stream and update latest_frame
        async def frame_producer():
            frame_count = 0
            try:
                async for frame_bytes in frame_stream:
                    if not processing_active:
                        break
                        
                    frame_count += 1
                    # Update latest frame immediately
                    # We store bytes to defer decoding to the consumer to keep producer fast
                    latest_frame["data"] = frame_bytes
                    latest_frame["timestamp"] = datetime.utcnow()
                    
                    # Update session frame count (lightweight)
                    # We might want to throttle this DB update or do it in consumer
                    # For now keep it here to track actual received frames
                    if frame_count % 30 == 0:  # Update every ~1s (assuming 30fps)
                        try:
                            await self.session_service.increment_frame_count(exam_session_id)
                        except Exception as e:
                            logger.warning(f"Failed to increment frame count: {e}")
                            
            except Exception as e:
                logger.error(f"Error in frame producer: {e}", exc_info=True)
            finally:
                nonlocal processing_active
                processing_active = False
        
        # Start producer
        producer_task = asyncio.create_task(frame_producer())
        
        # Consumer Loop: Periodically process the latest frame
        try:
            # Determine processing interval
            # If verification frequency is 10 (every 10 frames), and fps is 30, that's 3 times/sec.
            # But here frequency is simpler: how often do we WANT to verify per second?
            # Let's say we want 5Hz verification for responsiveness. 
            verification_interval = 0.2 # 5 times per second
            
            while processing_active:
                start_time = asyncio.get_event_loop().time()
                
                # Check if we have a frame
                if latest_frame["data"] is not None:
                    # Grab reference and clear (or just read latest)
                    current_frame_bytes = latest_frame["data"]
                    timestamp = latest_frame["timestamp"]
                    
                    # Avoid re-processing same frame if stream is slow
                    # We can clear it after reading to ensure unique processing
                    latest_frame["data"] = None 
                    
                    try:
                        # Decode
                        frame = self._decode_frame(current_frame_bytes)
                        
                        if frame is not None:
                            # Verify
                            alert = await self.continuous_verifier.verify_frame_continuous(
                                exam_session_id=exam_session_id,
                                candidate_id=candidate_id,
                                frame=frame,
                                timestamp=timestamp,
                                enrolled_embedding=enrolled_embedding
                            )
                            
                            if alert:
                                yield alert
                                
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                
                # Wait for next cycle
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0.01, verification_interval - elapsed)
                
                # Yield control to allow producer to run
                await asyncio.sleep(sleep_time)
                
                # Check if producer finished
                if producer_task.done():
                    break
                    
        except Exception as e:
            logger.error(f"Error in frame consumer: {e}", exc_info=True)
        finally:
            processing_active = False
            # Ensure producer task is cleaned up
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
    
    def _decode_frame(self, frame_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decode H.264 encoded frame to RGB numpy array using ffmpeg/av.
        
        Args:
            frame_bytes: Encoded frame bytes (H.264 NAL units or container format)
            
        Returns:
            Decoded frame as numpy array (BGR format for OpenCV)
        """
        try:
            # First, try to decode as image (JPEG/PNG) if gateway sends decoded frames
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                return frame
            
            # If that fails, try H.264 decoding using PyAV
            # Create a BytesIO container for the frame data
            container = av.open(BytesIO(frame_bytes), format='h264')
            
            # Try to read video frames
            for frame in container.decode(video=0):
                # Convert PyAV frame to numpy array
                frame_array = frame.to_ndarray(format='bgr24')
                container.close()
                return frame_array
            
            container.close()
            
            # If H.264 decoding fails, try raw RGB format (fallback)
            # Assume raw RGB24 format: width * height * 3 bytes
            # This is a fallback - in production, gateway should send proper format
            logger.warning("H.264 decoding failed, attempting raw RGB format")
            
            # Try common resolutions (this is a fallback, should be known from metadata)
            for width, height in [(640, 480), (1280, 720), (1920, 1080)]:
                expected_size = width * height * 3
                if len(frame_bytes) == expected_size:
                    frame = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = frame.reshape((height, width, 3))
                    return frame
            
            logger.error(f"Failed to decode frame: unknown format, size={len(frame_bytes)}")
            return None
            
        except av.AVError as e:
            logger.warning(f"PyAV decode error (may not be H.264): {e}")
            # Fallback to OpenCV image decode
            try:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return frame
            except Exception as e2:
                logger.error(f"Fallback decode also failed: {e2}")
                return None
        except Exception as e:
            logger.error(f"Frame decode error: {e}", exc_info=True)
            return None
    
    async def _load_enrolled_embedding(
        self,
        candidate_id: str,
        db_session
    ) -> Optional[np.ndarray]:
        """
        Load enrolled embedding for a candidate.
        
        Args:
            candidate_id: Candidate identifier
            db_session: Database session
            
        Returns:
            Enrolled embedding as numpy array or None if not found
        """
        # Check cache first
        if candidate_id in self._enrolled_embeddings_cache:
            return self._enrolled_embeddings_cache[candidate_id]
        
        try:
            # Query database for enrollment
            result = await db_session.execute(
                select(FaceEnrollment)
                .where(FaceEnrollment.candidate_id == candidate_id)
                .order_by(FaceEnrollment.created_at.desc())
                .limit(1)
            )
            enrollment = result.scalar_one_or_none()
            
            if enrollment is None:
                logger.error(f"No enrollment found for candidate {candidate_id}")
                return None
            
            # Convert vector to numpy array
            embedding = np.array(enrollment.embedding, dtype=np.float32)
            
            # Cache the embedding
            self._enrolled_embeddings_cache[candidate_id] = embedding
            
            logger.info(f"Loaded enrolled embedding for candidate {candidate_id}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error loading enrolled embedding: {e}", exc_info=True)
            return None
    
    def clear_embedding_cache(self, candidate_id: Optional[str] = None):
        """
        Clear embedding cache.
        
        Args:
            candidate_id: Optional candidate ID to clear, or None to clear all
        """
        if candidate_id:
            self._enrolled_embeddings_cache.pop(candidate_id, None)
        else:
            self._enrolled_embeddings_cache.clear()
    
    async def get_frame_buffer(
        self,
        exam_session_id: str,
        window_seconds: Optional[int] = None
    ):
        """
        Get recent frames from buffer.
        
        Args:
            exam_session_id: Exam session identifier
            window_seconds: Time window in seconds
            
        Returns:
            List of Frame objects
        """
        return await self.frame_buffer.get_recent_frames(
            exam_session_id,
            window_seconds
        )
    
    async def close(self):
        """Close connections and cleanup."""
        await self.frame_buffer.close()

