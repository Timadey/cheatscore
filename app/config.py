"""
Configuration management for SD Proctor service.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Tuple


class Settings(BaseSettings):
    # =========================
    # Application
    # =========================
    app_name: str = Field(default="SD Proctor")
    app_version: str = Field(default="0.1.0")
    app_url: str = Field(default="http://localhost:8000")
    debug: bool = Field(default=False)

    # =========================
    # Database
    # =========================
    database_url: str

    # =========================
    # Redis
    # =========================
    redis_url: str

    # =========================
    # Kafka
    # =========================
    kafka_bootstrap_servers: str
    kafka_alert_topic: str = "proctoring.alerts.v1"
    kafka_dlq_topic: str = "proctoring.alerts.dlq"
    kafka_max_retries: int = 3

    # =========================
    # AI / Model Configuration
    # =========================
    ai_model_device: str = "cpu"  # cpu | cuda
    ai_model_batch_size: int = 8

    face_match_threshold: float = 0.75
    face_detection_confidence: float = 0.5

    # Face Detection
    face_detection_model: str = "buffalo_sc"
    face_detection_model_path: str

    face_detection_frame_skip: int = 2

    # Face Verification
    face_verification_model: str = "buffalo_sc"
    face_verification_model_path: str
    embedding_dim: int = 512

    # =========================
    # Quality Checks
    # =========================
    min_face_size: int = 80
    min_sharpness: float = 100.0
    min_brightness: int = 40
    max_brightness: int = 220
    max_pose_deviation: float = 20.0

    # =========================
    # Gaze Tracking
    # =========================
    gaze_pitch_range: str = "-15,25"
    gaze_yaw_range: str = "-30,30"
    gaze_off_screen_duration: float = 4.0

    # =========================
    # WebRTC
    # =========================
    webrtc_stun_servers: str
    webrtc_turn_servers: str = ""
    webrtc_frame_buffer_size: int = 150
    webrtc_verification_frequency: int = 10
    webrtc_alert_websocket_path: str = "/api/v1/session/ws"

    # =========================
    # Alert Dispatcher
    # =========================
    alert_grpc_endpoint: str = ""
    alert_webhook_url: str = ""
    alert_retry_attempts: int = 5
    alert_retry_backoff: str = "100,500,2000,5000,10000"

    # =========================
    # Security
    # =========================
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # =========================
    # Monitoring
    # =========================
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    jaeger_enabled: bool = False
    jaeger_endpoint: str = ""

    # =========================
    # Data Retention
    # =========================
    embedding_retention_days: int = 90
    frame_buffer_retention_seconds: int = 5
    evidence_retention_days: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    # =========================
    # Derived Properties
    # =========================
    @property
    def kafka_servers_list(self) -> List[str]:
        return [s.strip() for s in self.kafka_bootstrap_servers.split(",")]

    @property
    def gaze_pitch_range_tuple(self) -> Tuple[float, float]:
        lo, hi = self.gaze_pitch_range.split(",")
        return float(lo), float(hi)

    @property
    def gaze_yaw_range_tuple(self) -> Tuple[float, float]:
        lo, hi = self.gaze_yaw_range.split(",")
        return float(lo), float(hi)

    @property
    def retry_backoff_list(self) -> List[int]:
        return [int(x.strip()) for x in self.alert_retry_backoff.split(",")]


settings = Settings()
