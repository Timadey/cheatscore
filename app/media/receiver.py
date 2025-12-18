"""
Video receiver for WebRTC tracks.
"""
import asyncio
import logging
import time
from datetime import datetime
from aiortc import MediaStreamTrack
from av import VideoFrame

from app.media.sampler import FrameSampler
from app.processing.pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)

class VideoReceiver:
    """
    Consumes frames from a WebRTC RemoteStreamTrack, samples them,
    and pushes to the processing pipeline.
    """
    def __init__(self, track: MediaStreamTrack, session_id: str, candidate_id: str, 
                 pipeline: ProcessingPipeline, pc, enrolled_embedding=None):
        self.track = track
        self.session_id = session_id
        self.candidate_id = candidate_id
        self.pipeline = pipeline
        self.pc = pc # Added pc argument
        self.enrolled_embedding = enrolled_embedding
        self.sampler = FrameSampler(target_fps=5.0) # Configurable
        self._task = None

    def start(self):
        logger.info("Starting video receiver")
        self._task = asyncio.create_task(self._consume())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _consume(self):
        try:
            logger.info(f"Starting video consumer for session {self.session_id}")
            while True:
                try:
                    frame: VideoFrame = await asyncio.wait_for(self.track.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(f"No frames received for 2s. Connection state: {self.pc.connectionState}")
                    continue

                # Check sampler
                if self.sampler.should_process(time.time()):
                    # Convert to BGR numpy array
                    img = frame.to_ndarray(format="bgr24")
                    # print("Received frame:", img.shape, "for session", self.session_id)
                    
                    # Prepare data
                    data = {
                        "exam_session_id": self.session_id,
                        "candidate_id": self.candidate_id,
                        "frame": img,
                        "timestamp": datetime.utcnow(),
                        "enrolled_embedding": self.enrolled_embedding
                    }
                    
                    # Enqueue (fire and forget)
                    await self.pipeline.enqueue_frame(data)
                    
        except Exception as e:
            print("Video receiver for session", self.session_id, "stopped:", e)
