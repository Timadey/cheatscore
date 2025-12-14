"""
Continuous verification worker for real-time face verification during exam sessions.
"""
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from app.inference.face_detection import FaceDetector
from app.inference.face_verification import FaceVerifier
from app.services.verification_service import VerificationService
from app.schemas import AlertEvent, EventType, AlertEvidence
from app.config import settings

logger = logging.getLogger(__name__)


class ContinuousVerifier:
    """Continuous face verification during exam sessions."""

    def __init__(self):
        """Initialize continuous verifier."""
        self.face_detector = FaceDetector()
        self.face_verifier = FaceVerifier()
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
                "multiple_faces_duration": 0.0,
                "no_face_duration": 0.0,
                "last_face_detection_time": None
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
        return state["frame_count"] % settings.webrtc_verification_frequency == 0

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

        Args:
            exam_session_id: Exam session ID
            candidate_id: Candidate ID
            detections: List of detected faces
            state: Session state
            timestamp: Current timestamp

        Returns:
            AlertEvent if threshold exceeded, None otherwise
        """
        if len(detections) > 1:
            state["multiple_faces_duration"] += (1.0 / 30.0)  # Assume 30 FPS

            if state["multiple_faces_duration"] >= 3.0:  # 3 seconds
                return self._create_alert(
                    exam_session_id=exam_session_id,
                    candidate_id=candidate_id,
                    event_type=EventType.MULTIPLE_FACES,
                    severity_score=0.85,
                    evidence={
                        "face_count": len(detections),
                        "duration_seconds": state["multiple_faces_duration"]
                    },
                    timestamp=timestamp
                )
        else:
            state["multiple_faces_duration"] = 0.0

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

        Args:
            exam_session_id: Exam session ID
            candidate_id: Candidate ID
            state: Session state
            timestamp: Current timestamp

        Returns:
            AlertEvent if threshold exceeded, None otherwise
        """
        state["no_face_duration"] += 1
        state["last_face_detection_time"] = None

        logger.debug(f"No face detected. Duration: {state['no_face_duration']}s")

        if state["no_face_duration"] >= 5.0:  # 5 seconds
            logger.info("No face detected for prolonged period")
            return self._create_alert(
                exam_session_id=exam_session_id,
                candidate_id=candidate_id,
                event_type=EventType.FACE_NOT_FOUND,
                severity_score=0.90,
                evidence={
                    "duration_seconds": state["no_face_duration"]
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
            current_embedding = self.face_verifier.extract_embedding_from_frame(
                frame,
                detection["bbox"],
                detection.get("landmarks")
            )

            # Compute similarity
            similarity = self.face_verifier.compute_similarity(
                current_embedding,
                enrolled_embedding
            )

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
                state["no_face_duration"] = 0.0
                state["last_face_detection_time"] = timestamp

            # Verify face if exactly one detected
            if len(detections) == 1:
                detection = detections[0]
                threshold = threshold or settings.face_match_threshold

                return self._handle_face_verification(
                    exam_session_id,
                    candidate_id,
                    detection,
                    frame,
                    enrolled_embedding,
                    threshold,
                    state,
                    timestamp
                )

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
            del self.session_state[exam_session_id]