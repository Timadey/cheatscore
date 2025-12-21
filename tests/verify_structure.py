"""
Verification script for WebRTC structure.
"""
import sys
import os
import asyncio
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

# Mock dependencies that might require running services (Redis, Postgres)
sys.modules['app.utils.db'] = MagicMock()
sys.modules['app.utils.redis_client'] = MagicMock()
sys.modules['app.config'] = MagicMock()

async def verify():
    print("Verifying imports...")
    try:
        from app.webrtc.signaling.webrtc import WebRTCManager
        from app.webrtc.media.sampler import FrameSampler
        from app.webrtc.processing.pipeline import ProcessingPipeline
        from app.webrtc.media.receiver import VideoReceiver
        print("Imports successful.")
    except Exception as e:
        print(f"Import failed: {e}")
        return

    print("Verifying Manager Singleton...")
    manager = WebRTCManager.get_instance()
    assert manager is not None
    print("Manager verified.")

    print("Verifying Sampler...")
    sampler = FrameSampler(target_fps=5.0)
    assert sampler.should_process(0) == True
    assert sampler.should_process(0.01) == False
    print("Sampler verified.")

    print("Verification complete.")

if __name__ == "__main__":
    asyncio.run(verify())
