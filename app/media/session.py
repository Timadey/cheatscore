"""
Media session state.
"""
from typing import Optional
from aiortc import RTCPeerConnection, RTCSessionDescription
from app.media.receiver import VideoReceiver
from app.processing.pipeline import ProcessingPipeline
from app.alerts.dispatcher import DataChannelAlertDispatcher

class MediaSession:
    """
    Holds the state for a single WebRTC session.
    """
    def __init__(self, exam_session_id: str, candidate_id: str, pc: RTCPeerConnection, enrolled_embedding=None):
        self.exam_session_id = exam_session_id
        self.candidate_id = candidate_id
        self.pc = pc
        self.enrolled_embedding = enrolled_embedding
        
        # Components
        # We need a reference to the global/shared pipeline or create one per session?
        # Creating one per session allows for easier cleanup, but might be resource heavy?
        # The pipeline has a queue and a worker. A shared pipeline with a shared worker pool is better.
        # But for now, let's create a pipeline per session to keep it isolated as requested "multiple sessions safely".
        # Wait, if we spawn a thread pool for models, we should reuse that. 
        # The pipeline class uses `FaceModelManager` singleton, so the model is shared.
        # The pipeline itself just runs an asyncio loop.
        # We can have one pipeline per session.
        self.alert_dispatcher_impl = DataChannelAlertDispatcher(self) # Takes manager-like interface
        self.pipeline = ProcessingPipeline(self.alert_dispatcher_impl)
        
        self.receiver: Optional[VideoReceiver] = None
        self.data_channel = None

    async def start(self):
        await self.pipeline.start()

    async def stop(self):
        if self.receiver:
            await self.receiver.stop()
        await self.pipeline.stop()
        await self.pc.close()

    # AlertDispatcher expects a `send_data` method on the manager/session
    async def send_data(self, session_id: str, message: str):
        if self.data_channel and self.data_channel.readyState == "open":
            self.data_channel.send(message)
