"""
WebRTC Manager.
"""
import logging
import asyncio
from typing import Dict, Optional
import numpy as np

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from sqlalchemy import select

from app.config import settings
from app.media.session import MediaSession
from app.media.receiver import VideoReceiver
from app.models import FaceEnrollment
from app.services.session_service import SessionService
from app.utils.db import AsyncSessionLocal

logger = logging.getLogger(__name__)

class WebRTCManager:
    """
    Singleton manager for WebRTC sessions.
    """
    _instance = None
    
    def __init__(self):
        self.sessions: Dict[str, MediaSession] = {}
        self.session_service = SessionService()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def create_session(self, exam_session_id: str, candidate_id: str, sdp_offer: str, type: str) -> str:
        """
        Create a new WebRTC session.
        Returns SDP answer.
        """
        # 1. Load enrolled embedding
        enrolled_embedding = await self._load_enrolled_embedding(candidate_id)
        if enrolled_embedding is None:
            logger.warning(f"No enrolled embedding for {candidate_id}, continuing without verification reference.")
        
        # 2. Create PC with ICE Config
        # Parse ICE servers from config
        ice_servers_config = []
        try:
            import json
            # If settings.webrtc_turn_servers is a JSON string of servers
            if settings.webrtc_turn_servers:
                 servers = json.loads(settings.webrtc_turn_servers)
                 for s in servers:
                     ice_servers_config.append(RTCIceServer(**s))
        except:
             # Fallback or simple format
             pass
        
        # Always add STUN
        ice_servers_config.append(RTCIceServer(urls=settings.webrtc_stun_servers.split(",")))

        config = RTCConfiguration(iceServers=ice_servers_config)
        pc = RTCPeerConnection(configuration=config)
        
        # 3. Create Session Object
        session = MediaSession(exam_session_id, candidate_id, pc, enrolled_embedding)
        self.sessions[exam_session_id] = session
        
        # 4. Setup Event Handlers
        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"Data channel opened for session {exam_session_id}")
            session.data_channel = channel

        @pc.on("track")
        def on_track(track):
            logger.info(f"Track received: {track.kind} for session {exam_session_id}")
            if track.kind == "video":
                # Create Receiver
                receiver = VideoReceiver(
                    track=track,
                    session_id=exam_session_id,
                    candidate_id=candidate_id,
                    pc=pc,
                    pipeline=session.pipeline,
                    enrolled_embedding=enrolled_embedding
                )
                session.receiver = receiver
                receiver.start()
            
            @track.on("ended")
            async def on_ended():
                logger.info(f"Track ended for session {exam_session_id}")
                await session.stop()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state change: {pc.connectionState} for session {exam_session_id}")
            if pc.connectionState in ["failed", "closed"]:
                await self.close_session(exam_session_id)

        # 5. Handle Offer
        offer = RTCSessionDescription(sdp=sdp_offer, type=type)
        await pc.setRemoteDescription(offer)
        
        # 6. Create Answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        logger.info("ICE configuration: " + str(config))
        logger.info("ICE Gathering State: " + pc.iceGatheringState)
        
        # Wait for ICE gathering to complete (Vanilla ICE)
        # This ensures the SDP contains the TURN candidates
        while pc.iceGatheringState != "complete":
             await asyncio.sleep(0.1)
             # Timeout safety? We rely on aiortc finding it reasonably fast.
        
        # 7. Start Pipeline
        await session.start()
        
        return pc.localDescription.sdp

    async def close_session(self, exam_session_id: str):
        if exam_session_id in self.sessions:
            session = self.sessions[exam_session_id]
            await session.stop()
            del self.sessions[exam_session_id]
            print("Closed session", exam_session_id)
            logger.info(f"Closed session {exam_session_id}")

    async def _load_enrolled_embedding(self, candidate_id: str) -> Optional[np.ndarray]:
        async with AsyncSessionLocal() as db:
            try:
                result = await db.execute(
                    select(FaceEnrollment)
                    .where(FaceEnrollment.candidate_id == candidate_id)
                    .order_by(FaceEnrollment.created_at.desc())
                    .limit(1)
                )
                enrollment = result.scalar_one_or_none()
                if enrollment:
                    return enrollment.embedding
                     # return np.array(enrollment.embedding, dtype=np.float32)
            except Exception as e:
                logger.error(f"Error loading embedding: {e}")
        return None
