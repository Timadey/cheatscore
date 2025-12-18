"""
Continuous verification worker for real-time face verification during exam sessions.
"""
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from app.inference.face_detection import face_detector_loaded
from app.inference.face_verification import face_verifier_loaded
# from app.inference.attention_analysis import AttentionAnalyzer, AttentionResult, GazeDirection
from app.services.verification_service import VerificationService
from app.schemas import AlertEvent, EventType, AlertEvidence
from app.config import settings

logger = logging.getLogger(__name__)


class ContinuousVerifier:
    """Continuous face verification during exam sessions."""

    def __init__(self):
        """Initialize continuous verifier."""
        self.face_detector = face_detector_loaded
        self.face_verifier = face_verifier_loaded
        self.verification_service = VerificationService()

        # Track state per session
        self.session_state: Dict[str, Dict[str, Any]] = {}

    def _get_session_state(self, exam_session_id: str) -> Dict[str, Any]:
        """Get or create session state."""
        if exam_session_id not in self.session_state:
            self.session_state[exam_session_id] = {
                "frame_count": 0,
                "verification_count": 0,
                "consecutive_mismatches": 0,
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
        return self.session_state[exam_session_id]

    def _should_verify_frame(self, state: Dict[str, Any]) -> bool:
        """
        Determine if current frame should be verified.
        
        Args:
            state: Session state dictionary
            
        Returns:
            True if frame should be verified
        """
        # With decoupled processing, we verify every frame we actually process
        # The logic for skipping frames is now handled by the GatewayAdapter consumer
        return True

    def _handle_multiple_faces(
        self,
        exam_session_id: str,
        candidate_id: str,
        detections: list,
        state: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[AlertEvent]:
        """
        Handle multiple faces detection.
        """
        if len(detections) > 1:
            if state["multiple_faces_start_time"] is None:
                state["multiple_faces_start_time"] = timestamp
            
            duration = (timestamp - state["multiple_faces_start_time"]).total_seconds()
            
            if duration >= 3.0:  # 3 seconds
                # Reset timer to avoid flooding alerts for the same event
                # Or keep it set if we want to alert periodically? 
                # For now, let's reset to send another alert after 3 more seconds
                state["multiple_faces_start_time"] = timestamp
                
                return self._create_alert(
                    exam_session_id=exam_session_id,
                    candidate_id=candidate_id,
                    event_type=EventType.MULTIPLE_FACES,
                    severity_score=0.85,
                    evidence={
                        "face_count": len(detections),
                        "duration_seconds": duration
                    },
                    timestamp=timestamp
                )
        else:
            state["multiple_faces_start_time"] = None
            
        return None

    def _handle_no_face(
        self,
        exam_session_id: str,
        candidate_id: str,
        state: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[AlertEvent]:
        """
        Handle no face detection.
        """
        if state["no_face_start_time"] is None:
            state["no_face_start_time"] = timestamp
            
        duration = (timestamp - state["no_face_start_time"]).total_seconds()
        state["last_face_detection_time"] = None
        
        logger.debug(f"No face detected. Duration: {duration:.2f}s")
        
        if duration >= 5.0:  # 5 seconds
            logger.info("No face detected for prolonged period")
            # Reset timer to create periodic alerts if condition persists
            state["no_face_start_time"] = timestamp
            
            return self._create_alert(
                exam_session_id=exam_session_id,
                candidate_id=candidate_id,
                event_type=EventType.FACE_NOT_FOUND,
                severity_score=0.90,
                evidence={
                    "duration_seconds": duration
                },
                timestamp=timestamp
            )
            
        return None

    def _handle_face_verification(
        self,
        exam_session_id: str,
        candidate_id: str,
        detection: Dict[str, Any],
        frame: np.ndarray,
        enrolled_embedding: np.ndarray,
        threshold: float,
        state: Dict[str, Any],
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

            # current_embedding = self.face_verifier.extract_embedding_from_frame(
            #     frame,
            #     detection["bbox"],
            #     detection.get("landmarks")
            # )

            # Compute similarity
            similarity = self.face_verifier.compute_similarity(
                current_embedding,
                enrolled_embedding
            )
            # print(f"🔥 SIMILARITY: {similarity} 🔥")
            # Check for mismatch
            if similarity < threshold:
                state["consecutive_mismatches"] += 1

                if state["consecutive_mismatches"] >= 3:
                    return self._create_alert(
                        exam_session_id=exam_session_id,
                        candidate_id=candidate_id,
                        event_type=EventType.FACE_MISMATCH,
                        severity_score=0.95,
                        evidence={
                            "similarity_score": similarity,
                            "threshold": threshold,
                            "consecutive_mismatches": state["consecutive_mismatches"],
                            "confidence_metrics": {
                                "face_sim": similarity
                            }
                        },
                        timestamp=timestamp
                    )
            else:
                # Reset mismatch counter on successful match
                state["consecutive_mismatches"] = 0

            state["verification_count"] += 1
            state["last_verification_time"] = timestamp

        except Exception as e:
            logger.warning(f"Failed to extract embedding or compute similarity: {e}")
            # Don't create an alert for embedding extraction failures

        return None

    async def verify_frame_continuous(
        self,
        exam_session_id: str,
        candidate_id: str,
        frame: np.ndarray,
        timestamp: datetime,
        enrolled_embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> Optional[AlertEvent]:
        """
        Verify frame and return alert if anomaly detected.

        Args:
            exam_session_id: Exam session identifier
            candidate_id: Candidate identifier
            frame: Frame image (BGR format)
            timestamp: Frame timestamp
            enrolled_embedding: Enrolled face embedding
            threshold: Similarity threshold

        Returns:
            AlertEvent if anomaly detected, None otherwise
        """
        state = self._get_session_state(exam_session_id)
        state["frame_count"] += 1

        # Check if we should verify this frame
        if not self._should_verify_frame(state):
            return None

        logger.debug(f"Continuous verification on frame: {state['frame_count']}")

        try:
            # Detect faces using InsightFace
            detections = self.face_detector.detect(frame)
            logger.debug(f"Detected {len(detections)} faces")

            # Check for multiple faces
            alert = self._handle_multiple_faces(
                exam_session_id,
                candidate_id,
                detections,
                state,
                timestamp
            )
            if alert:
                return alert

            # Check for no face
            if len(detections) == 0:
                logger.debug("No face detected")
                return self._handle_no_face(
                    exam_session_id,
                    candidate_id,
                    state,
                    timestamp
                )
            else:
                state["no_face_start_time"] = None
                state["last_face_detection_time"] = timestamp

            # Verify face if exactly one detected
            if len(detections) == 1:
                detection = detections[0]
                threshold = threshold or settings.face_match_threshold

                # 1. Identity Verification (InsightFace)
                identity_alert = self._handle_face_verification(
                    exam_session_id,
                    candidate_id,
                    detection,
                    frame,
                    enrolled_embedding,
                    threshold,
                    state,
                    timestamp
                )
                if identity_alert:
                    return identity_alert
                    
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

            return None

        except Exception as e:
            logger.error(f"Error in continuous verification: {e}", exc_info=True)
            return None

    def _create_alert(
        self,
        exam_session_id: str,
        candidate_id: str,
        event_type: EventType,
        severity_score: float,
        evidence: Dict[str, Any],
        timestamp: datetime
    ) -> AlertEvent:
        """Create an alert event."""
        return AlertEvent(
            exam_session_id=exam_session_id,
            candidate_id=candidate_id,
            timestamp=timestamp,
            event_type=event_type,
            severity_score=severity_score,
            ai_model_source=f"{self.face_verifier.model_name or 'insightface'}:v1.0",
            evidence=AlertEvidence(**evidence),
            metadata={
                "verification_count": self.session_state.get(
                    exam_session_id, {}
                ).get("verification_count", 0)
            }
        )

    def reset_session_state(self, exam_session_id: str):
        """Reset state for a session."""
        if exam_session_id in self.session_state:
            state = self.session_state[exam_session_id]
            # Close MediaPipe resources
            if "attention_analyzer" in state:
                state["attention_analyzer"].close()
            del self.session_state[exam_session_id]

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
    #     state["last_attention_check_time"] = timestamp
    #     analyzer: AttentionAnalyzer = state["attention_analyzer"]
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
    #         if state["gaze_off_screen_start_time"] is None:
    #             state["gaze_off_screen_start_time"] = timestamp
    #             state["gaze_off_screen_direction"] = result.gaze_direction.value
    #             logger.debug(f"Gaze deviation started: {result.gaze_direction.value}")
    #
    #         # Check duration
    #         duration = (timestamp - state["gaze_off_screen_start_time"]).total_seconds()
    #
    #         if duration >= settings.gaze_off_screen_threshold:
    #             # PATTERN DETECTED: Sustained Gaze off-screen
    #
    #             # Reset timer to avoid spam (or create one alert per episode)
    #             # We reset to current time to re-trigger if it persists for ANOTHER threshold duration
    #             state["gaze_off_screen_start_time"] = timestamp
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
    #         if state["gaze_off_screen_start_time"] is not None:
    #             duration = (timestamp - state["gaze_off_screen_start_time"]).total_seconds()
    #             if duration > 0.5:
    #                  logger.debug(f"Gaze returned to screen after {duration:.1f}s")
    #         state["gaze_off_screen_start_time"] = None
    #         state["gaze_off_screen_direction"] = None
    #
    #     return None