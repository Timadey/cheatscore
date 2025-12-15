"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


# Enums
class EventType(str, Enum):
    """Alert event types."""
    FACE_MISMATCH = "FACE_MISMATCH"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    FACE_NOT_FOUND = "FACE_NOT_FOUND"
    EYE_OFF_SCREEN = "EYE_OFF_SCREEN"
    GAZE_OFF_SCREEN = "GAZE_OFF_SCREEN"
    AUDIO_ANOMALY = "AUDIO_ANOMALY"
    SUSPICIOUS_MOVEMENT = "SUSPICIOUS_MOVEMENT"
    LEFT_FRAME = "LEFT_FRAME"
    PRESENCE_LOSS = "PRESENCE_LOSS"
    MULTIPLE_PERSONS = "MULTIPLE_PERSONS"
    VOICE_DETECTED = "VOICE_DETECTED"
    MULTIPLE_SPEAKERS = "MULTIPLE_SPEAKERS"


class SeverityLevel(str, Enum):
    """Severity levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertSensitivity(str, Enum):
    """Alert sensitivity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Enrollment Schemas
class ImageData(BaseModel):
    """Single image data for enrollment."""
    image_base64: str = Field(..., description="Base64-encoded image")
    capture_timestamp: Optional[datetime] = None
    device_info: Optional[Dict[str, Any]] = None


class EnrollmentRequest(BaseModel):
    """Request for face enrollment."""
    candidate_id: str = Field(..., description="Unique candidate identifier")
    images: List[ImageData] = Field(..., min_items=1, max_items=10, description="3-5 images recommended")
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingInfo(BaseModel):
    """Information about a face embedding."""
    embedding_id: str
    quality_score: float = Field(..., ge=0.0, le=1.0)
    face_bbox: Optional[List[int]] = None
    pose_angles: Optional[Dict[str, float]] = None
    sharpness_score: Optional[float] = None
    face_image: Optional[str] = None  # Base64 encoded face crop
    selected: bool = False


class EnrollmentResponse(BaseModel):
    """Response from enrollment endpoint."""
    status: str = "success"
    candidate_id: str
    enrollment_id: str
    embeddings: List[EmbeddingInfo]
    average_embedding_stored: bool
    enrollment_timestamp: datetime
    recommendations: Optional[List[str]] = None


# Verification Schemas
class VerificationRequest(BaseModel):
    """Request for face verification."""
    candidate_id: str = Field(..., description="Candidate identifier")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image")
    frame_id: Optional[str] = Field(None, description="Reference to buffered frame")
    timestamp: Optional[datetime] = None
    trigger: Optional[str] = None


class FaceQuality(BaseModel):
    """Face quality metrics."""
    sharpness: float
    pose_deviation: float  # degrees from frontal
    lighting_quality: float


class VerificationResponse(BaseModel):
    """Response from verification endpoint."""
    match: bool
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    threshold_used: float
    comparison_embedding_id: Optional[str] = None
    face_quality: Optional[FaceQuality] = None
    timestamp: datetime


# Session Schemas
class VerificationPolicy(BaseModel):
    """Verification policy configuration."""
    face_match_threshold: float = 0.75
    gaze_tolerance_degrees: float = 30.0
    max_offline_seconds: float = 10.0
    alert_sensitivity: AlertSensitivity = AlertSensitivity.MEDIUM


class FeaturesEnabled(BaseModel):
    """Feature flags for session."""
    face_verification: bool = True
    gaze_tracking: bool = True
    pose_analysis: bool = True
    audio_monitoring: bool = True
    speech_transcription: bool = False


class SessionStartRequest(BaseModel):
    """Request to start an exam session."""
    exam_session_id: str
    candidate_id: str
    exam_id: Optional[str] = None
    frontend_instance_id: Optional[str] = None
    verification_policy: Optional[VerificationPolicy] = None
    features_enabled: Optional[FeaturesEnabled] = None
    metadata: Optional[Dict[str, Any]] = None


class ICEServer(BaseModel):
    """WebRTC ICE server configuration."""
    urls: str
    username: Optional[str] = None
    credential: Optional[str] = None


class VideoConstraints(BaseModel):
    """Video constraints for WebRTC."""
    width: Optional[Dict[str, int]] = None
    height: Optional[Dict[str, int]] = None
    frameRate: Optional[Dict[str, int]] = None


class AudioConstraints(BaseModel):
    """Audio constraints for WebRTC."""
    sampleRate: int = 16000
    channelCount: int = 1
    echoCancellation: bool = True


class WebRTCConfig(BaseModel):
    """WebRTC configuration."""
    ice_servers: List[ICEServer]
    video_constraints: Optional[VideoConstraints] = None
    audio_constraints: Optional[AudioConstraints] = None


class SessionStartResponse(BaseModel):
    """Response from session start endpoint."""
    status: str = "active"
    session_id: str
    websocket_url: Optional[str] = None
    webrtc_config: Optional[WebRTCConfig] = None
    calibration_required: bool = True
    calibration_duration_seconds: int = 30
    session_expires_at: Optional[datetime] = None


# Alert Schemas
class AlertEvidence(BaseModel):
    """Evidence attached to an alert."""
    frame_ids: Optional[List[str]] = None
    thumbnail: Optional[str] = None
    audio_snippet_id: Optional[str] = None
    gaze_vector: Optional[List[float]] = None
    pose_keypoints: Optional[List[List[float]]] = None
    transcript_snippet: Optional[str] = None
    confidence_metrics: Optional[Dict[str, float]] = None


class AlertEvent(BaseModel):
    """Canonical alert event structure."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    exam_session_id: str
    candidate_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: EventType
    severity_score: float = Field(..., ge=0.0, le=1.0)
    ai_model_source: str
    evidence: AlertEvidence = Field(default_factory=AlertEvidence)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WSAlertMessage(BaseModel):
    """WebSocket alert message format."""
    type: str = "alert"
    event_id: str
    exam_session_id: str
    event_type: EventType
    severity_score: float
    timestamp: datetime
    message: str
    action_required: str = "warning"  # "warning", "escalate", "auto-pause"
    evidence_thumbnail: Optional[str] = None
    dismiss_after_seconds: Optional[int] = 10

