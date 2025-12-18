"""
Frame sampling logic.
"""
import time

class FrameSampler:
    """
    Decides whether to process a frame based on FPS target.
    """
    def __init__(self, target_fps: float = 5.0):
        self.target_interval = 1.0 / target_fps
        self.last_processed_time = 0.0

    def should_process(self, timestamp: float) -> bool:
        """
        Check if frame at timestamp should be processed.
        Using system monotonic time for simplicity in interval checking.
        """
        # Note: timestamp arg here is usually presentation timestamp.
        # But to ensure consistent wall-clock FPS, we can check current time.
        now = time.monotonic()
        if now - self.last_processed_time >= self.target_interval:
            self.last_processed_time = now
            return True
        return False
