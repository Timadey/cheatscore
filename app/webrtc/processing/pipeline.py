"""
Async processing pipeline for ML inference.
Consumes sampled frames and runs detection/verification models.
"""
import asyncio
import logging
from typing import Optional

from app.inference.face_model_manager import FaceModelManager
from app.webrtc.processing.continuous_verifier import ContinuousVerifier
from app.alerts.datachannel_dispatcher import DataChannelAlertDispatcher

logger = logging.getLogger(__name__)

class ProcessingPipeline:
    """
    Consumer pipeline that processes frames from the queue.
    """
    def __init__(self, alert_dispatcher: DataChannelAlertDispatcher, verifier: ContinuousVerifier):
        self.queue = asyncio.Queue(maxsize=1)
        self.running = False
        self.alert_dispatcher = alert_dispatcher
        self.verifier = verifier
        # Initialize model manager (ensure it's loaded)
        self.model_manager = FaceModelManager.get_instance()
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the processing worker."""
        if self.running:
            return
        
        # Ensure models are initialized
        if not self.model_manager.is_initialized:
            self.model_manager.initialize()

        self.running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("Processing pipeline started")

    async def stop(self):
        """Stop the processing worker."""
        self.running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Processing pipeline stopped")

    async def enqueue_frame(self, frame_data: dict):
        """
        Push a frame to the processing queue.
        frame_data: {
          "exam_session_id": str,
          "candidate_id": str,
          "frame": np.ndarray,
          "timestamp": datetime
        }
        """
        if not self.running:
            return
            
        try:
            # Non-blocking put; if queue is full, drop the old frame and put the new one
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            
            self.queue.put_nowait(frame_data)
        except Exception as e:
            logger.warning(f"Failed to enqueue frame for session {frame_data.get('exam_session_id')}: {e}")

    async def _process_queue(self):
        """
        Continuous loop consuming frames.
        """
        while self.running:
            try:
                frame_data = await self.queue.get()
                
                # Process the frame
                await self._process_frame(frame_data)
                
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing pipeline: {e}", exc_info=True)

    async def _process_frame(self, data: dict):
        """
        Run inference on a single frame.
        """
        session_id = data["exam_session_id"]
        # candidate_id = data["candidate_id"]
        frame = data["frame"]
        timestamp = data["timestamp"]
        enrolled_embedding = data.get("enrolled_embedding")

        try:
             # Using ContinuousVerifier to verify frame
             # This reuses the existing logic which calls InsightFace
             # logger.info("Running continuous verification inference")
             alerts = self.verifier.verify_frame_continuous(
                 # exam_session_id=session_id,
                 # candidate_id=candidate_id,
                 frame=frame,
                 timestamp=timestamp,
                 enrolled_embedding=enrolled_embedding
             )

             async for alert in alerts:
                 await self.alert_dispatcher.dispatch(alert)

        except Exception as e:
            logger.error(f"Inference failed for session {session_id}: {e}")
