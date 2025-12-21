"""
Continuous verification worker for real-time face verification during exam sessions.
"""
import copy
import numpy as np
from typing import Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import logging
import os
import cv2

from app.inference.insightface.face_detection import face_detector_loaded
from app.inference.insightface.face_verification import face_verifier_loaded
# from app.inference.attention_analysis import AttentionAnalyzer, AttentionResult, GazeDirection
from app.schemas import AlertEvent, EventType, AlertEvidence
from app.config import settings


logger = logging.getLogger(__name__)

__INITIAL_SESSION_STATE__ = {
    "frame_count": 0,
    "verification_count": 0,
    "consecutive_mismatches": 0,
    # "multiple_faces_count": 0,
    "multiple_faces_total_duration": 0.0,
    "no_face_total_duration": 0.0,
    "face_mismatch_total_duration": 0.0,
    "face_mismatch_start_time": None,
    # Face Verification State
    "last_verification_time": None,
    "multiple_faces_start_time": None,
    "no_face_start_time": None,
    "last_face_detection_time": None,
    # Attention Tracking State
    # "attention_analyzer": AttentionAnalyzer(),
    "last_attention_check_time": None,
    "gaze_off_screen_start_time": None,
    "gaze_off_screen_direction": None,
    "head_pose_violation_start_time": None
}


class ContinuousVerifier:
    """Continuous face verification during exam sessions."""

    def __init__(self, exam_session_id: str, candidate_id: str):
        """Initialize continuous verifier."""
        self.face_detector = face_detector_loaded
        self.face_verifier = face_verifier_loaded
        self.exam_session_id = exam_session_id
        self.candidate_id = candidate_id

        # Track state per session
        self.session_state = copy.deepcopy(__INITIAL_SESSION_STATE__)

    def _get_session_state(self) -> Dict[str, Any]:
        """Get or create session state."""


    def _handle_multiple_faces(
        self,
        detections: list,
        timestamp: datetime
    ) -> Optional[AlertEvent]:
        """
        Handle multiple faces detection.
        """
        if len(detections) > 1:
            if self.session_state["multiple_faces_start_time"] is None:
                self.session_state["multiple_faces_start_time"] = timestamp
                return None

            duration = (timestamp - self.session_state["multiple_faces_start_time"]).total_seconds()

            if duration >= 2.0:  # 5 seconds
                # Reset timer to avoid flooding alerts for the same event
                # Or keep it set if we want to alert periodically?
                # For now, let's reset to send another alert after 4 more seconds
                self.session_state["multiple_faces_start_time"] = timestamp
                # self.session_state["multiple_faces_count"] += 1
                self.session_state["multiple_faces_total_duration"] += duration

                return self._create_alert(
                    event_type=EventType.MULTIPLE_FACES,
                    severity_score=0.85,
                    evidence={
                        "face_count": len(detections),
                        "duration_seconds": duration,
                        "total_duration": self.session_state["multiple_faces_total_duration"]
                    },
                    timestamp=timestamp
                )
        else:
            self.session_state["multiple_faces_start_time"] = None

        return None

    def _handle_no_face(self, timestamp: datetime) -> Optional[AlertEvent]:
        """
        Handle no face detection.
        """
        if self.session_state["no_face_start_time"] is None:
            # First frame with no face
            self.session_state["no_face_start_time"] = timestamp
            return None  # don’t alert yet

        duration = (timestamp - self.session_state["no_face_start_time"]).total_seconds()
        self.session_state["last_face_detection_time"] = None

        logger.debug(f"No face detected. Duration: {duration:.2f}s")

        threshold = 2.0  # or 5.0, whichever you choose
        if duration >= threshold:
            logger.info(f"No face detected for {duration:.2f}s")
            self.session_state["no_face_total_duration"] += duration
            # Reset start time so next alert triggers after another threshold period
            self.session_state["no_face_start_time"] = timestamp

            return self._create_alert(
                event_type=EventType.FACE_NOT_FOUND,
                severity_score=0.90,
                evidence={
                    "duration_seconds": duration,
                    "total_duration": self.session_state["no_face_total_duration"]
                },
                timestamp=timestamp
            )

        return None

    def _handle_face_verification(
        self,
        detection: Dict[str, Any],
        enrolled_embedding: np.ndarray,
        threshold: float,
        timestamp: datetime
    ) -> Optional[AlertEvent]:
        """
        Handle face verification for a single detection.

        Args:
            exam_session_id: Exam session ID
            candidate_id: Candidate ID
            detection: Face detection dictionary
            frame: Frame image
            enrolled_embedding: Enrolled face embedding
            threshold: Similarity threshold
            state: Session state
            timestamp: Current timestamp

        Returns:
            AlertEvent if mismatch detected, None otherwise
        """
        try:
            # Extract embedding using InsightFace
            face = detection.get("face")
            current_embedding = face.embedding

            # Compute similarity
            similarity = self.face_verifier.compute_similarity(current_embedding, enrolled_embedding)
            # print(f"🔥 SIMILARITY: {similarity} 🔥")
            # Check for mismatch
            if similarity < threshold:
                logger.debug(f"Face mismatch detected (similarity: {similarity:.2f})")
                if self.session_state["face_mismatch_start_time"] is None:
                    # First mismatch frame
                    self.session_state["face_mismatch_start_time"] = timestamp
                    self.session_state["consecutive_mismatches"] = 1
                    return None

                duration = (
                        timestamp - self.session_state["face_mismatch_start_time"]
                ).total_seconds()

                self.session_state["consecutive_mismatches"] += 1

                logger.debug(
                    f"Face mismatch. Duration: {duration:.2f}s | "
                    f"Consecutive: {self.session_state['consecutive_mismatches']}"
                )

                if (
                        self.session_state["consecutive_mismatches"] >= 3
                        # and duration >= 5.0
                ):
                    self.session_state["face_mismatch_total_duration"] += duration

                    # Reset to avoid alert spam
                    self.session_state["face_mismatch_start_time"] = timestamp
                    self.session_state["consecutive_mismatches"] = 0

                    return self._create_alert(
                        event_type=EventType.FACE_MISMATCH,
                        severity_score=0.95,
                        evidence={
                            "similarity_score": similarity,
                            "threshold": threshold,
                            "duration_seconds": duration,
                            "total_duration": self.session_state["face_mismatch_total_duration"],
                        },
                        timestamp=timestamp,
                    )
            else:
                # Successful match resets everything
                self.session_state["consecutive_mismatches"] = 0
                self.session_state["face_mismatch_start_time"] = None

            self.session_state["verification_count"] += 1
            self.session_state["last_verification_time"] = timestamp

        except Exception as e:
            logger.warning(f"Failed to extract embedding or compute similarity: {e}")
            # Don't create an alert for embedding extraction failures

        return None

    async def verify_frame_continuous(
        self,
        frame: np.ndarray,
        timestamp: datetime,
        enrolled_embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> AsyncGenerator[AlertEvent | None, Any]:
        """
        Verify frame and return alert if anomaly detected.

        Args:
            # exam_session_id: Exam session identifier
            # candidate_id: Candidate identifier
            frame: Frame image (BGR format)
            timestamp: Frame timestamp
            enrolled_embedding: Enrolled face embedding
            threshold: Similarity threshold

        Returns:
            AlertEvent if anomaly detected, None otherwise
        """
        self.session_state["frame_count"] += 1

        logger.debug(f"Continuous verification on frame: {self.session_state.get('frame_count')}")

        # Save frame for debugging
        debug_dir = "debug_frames"
        os.makedirs(debug_dir, exist_ok=True)
        filename = os.path.join(debug_dir, f"{self.exam_session_id}_one_face_{timestamp.strftime('%H%M%S_%f')}.jpg")
        cv2.imwrite(filename, frame)
        logger.debug(f"Saved one-face frame to {filename}")

        try:
            # Detect faces using InsightFace
            detections = self.face_detector.detect(frame)
            logger.debug(f"Detected {len(detections)} faces")

            # Check for multiple faces
            alert = self._handle_multiple_faces(detections, timestamp)
            if alert:
                yield alert

            # Check for no face
            if len(detections) == 0:
                logger.debug("No face detected")
                
                # Save frame for debugging
                # debug_dir = "debug_frames"
                # os.makedirs(debug_dir, exist_ok=True)
                # filename = os.path.join(debug_dir, f"{self.exam_session_id}_no_face_{timestamp.strftime('%H%M%S_%f')}.jpg")
                # cv2.imwrite(filename, frame)
                # logger.debug(f"Saved no-face frame to {filename}")

                yield self._handle_no_face(timestamp)
            else:
                self.session_state["no_face_start_time"] = None
                self.session_state["last_face_detection_time"] = timestamp

            # Verify face if exactly one detected
            if len(detections) == 1:
                detection = detections[0]
                threshold = threshold or settings.face_match_threshold

                # Save frame for debugging
                # debug_dir = "debug_frames"
                # os.makedirs(debug_dir, exist_ok=True)
                # filename = os.path.join(debug_dir, f"{self.exam_session_id}_one_face_{timestamp.strftime('%H%M%S_%f')}.jpg")
                # cv2.imwrite(filename, frame)
                # logger.debug(f"Saved one-face frame to {filename}")

                # 1. Identity Verification (InsightFace)
                identity_alert = self._handle_face_verification(
                    detection,
                    enrolled_embedding,
                    threshold,
                    timestamp
                )
                if identity_alert:
                    yield identity_alert
                    
                # 2. Attention Analysis (MediaPipe) - Throttled
                # # Run pattern-based attention checks
                # attention_alert = self._monitor_attention_patterns(
                #     exam_session_id,
                #     candidate_id,
                #     frame,
                #     state,
                #     timestamp
                # )
                # if attention_alert:
                #     return attention_alert

            return

        except Exception as e:
            logger.error(f"Error in continuous verification: {e}", exc_info=True)
            return

    def _create_alert(
        self,

        event_type: EventType,
        severity_score: float,
        evidence: Dict[str, Any],
        timestamp: datetime
    ) -> AlertEvent:
        """Create an alert event."""
        return AlertEvent(
            exam_session_id=self.exam_session_id,
            candidate_id=self.candidate_id,
            timestamp=timestamp,
            event_type=event_type,
            severity_score=severity_score,
            ai_model_source=f"{self.face_verifier.model_name or 'insightface'}:v1.0",
            evidence=AlertEvidence(**evidence),
            metadata={
                "verification_count": self.session_state.get(
                    self.exam_session_id, {}
                ).get("verification_count", 0)
            }
        )

    def reset_session_state(self, exam_session_id: str):
        """Reset state for a session."""
        if self.session_state:
            # Close MediaPipe resources
            if "attention_analyzer" in self.session_state:
                self.session_state["attention_analyzer"].close()
            self.session_state = {}

    # def _monitor_attention_patterns(
    #     self,
    #     exam_session_id: str,
    #     candidate_id: str,
    #     frame: np.ndarray,
    #     state: Dict[str, Any],
    #     timestamp: datetime
    # ) -> Optional[AlertEvent]:
    #     """
    #     Run attention analysis and check for prohibited patterns.
    #     """
    #     # Throttle: Check frame rate for attention (e.g., 4 FPS)
    #     last_check = state.get("last_attention_check_time")
    #     if last_check:
    #         delta = (timestamp - last_check).total_seconds()
    #         if delta < (1.0 / settings.attention_check_fps):
    #             return None
    #
    #     self.session_state["last_attention_check_time"] = timestamp
    #     analyzer: AttentionAnalyzer = self.session_state["attention_analyzer"]
    #
    #     # Run inference
    #     result = analyzer.process_frame(frame)
    #     if not result:
    #         return None
    #
    #     # --- Pattern Logic ---
    #
    #     # 1. Gaze Off-Screen Check
    #     if not result.is_looking_at_screen:
    #         # If this is the start of a deviation, record it
    #         if self.session_state["gaze_off_screen_start_time"] is None:
    #             self.session_state["gaze_off_screen_start_time"] = timestamp
    #             self.session_state["gaze_off_screen_direction"] = result.gaze_direction.value
    #             logger.debug(f"Gaze deviation started: {result.gaze_direction.value}")
    #
    #         # Check duration
    #         duration = (timestamp - self.session_state["gaze_off_screen_start_time"]).total_seconds()
    #
    #         if duration >= settings.gaze_off_screen_threshold:
    #             # PATTERN DETECTED: Sustained Gaze off-screen
    #
    #             # Reset timer to avoid spam (or create one alert per episode)
    #             # We reset to current time to re-trigger if it persists for ANOTHER threshold duration
    #             self.session_state["gaze_off_screen_start_time"] = timestamp
    #
    #             logger.info(f"Gaze alert emitted: {result.gaze_direction.value} for {duration:.1f}s")
    #
    #             return self._create_alert(
    #                 exam_session_id=exam_session_id,
    #                 candidate_id=candidate_id,
    #                 event_type=EventType.GAZE_OFF_SCREEN,
    #                 severity_score=0.75,
    #                 evidence={
    #                     "gaze_direction": result.gaze_direction.value,
    #                     "duration_seconds": duration,
    #                     "head_pose": result.head_pose,
    #                     "gaze_vector": [float(result.gaze_direction == GazeDirection.LEFT), float(result.gaze_direction == GazeDirection.RIGHT), float(result.gaze_direction == GazeDirection.UP)] # Simplified vector
    #                 },
    #                 timestamp=timestamp
    #             )
    #     else:
    #         # Reset if looking at screen
    #         if self.session_state["gaze_off_screen_start_time"] is not None:
    #             duration = (timestamp - self.session_state["gaze_off_screen_start_time"]).total_seconds()
    #             if duration > 0.5:
    #                  logger.debug(f"Gaze returned to screen after {duration:.1f}s")
    #         self.session_state["gaze_off_screen_start_time"] = None
    #         self.session_state["gaze_off_screen_direction"] = None
    #
    #     return None