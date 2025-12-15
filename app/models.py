"""
SQLAlchemy database models for SD Proctor service.
"""
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid
import logging

from app.utils.vector_db import VectorDB
from app.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

# Try to use pgvector, fallback to JSON if not available
try:
    VectorType = VectorDB.get_vector_type(settings.embedding_dim)
    USE_VECTOR_DB = True
    logger.info("Using pgvector for embedding storage")
except ImportError:
    USE_VECTOR_DB = False
    VectorType = JSON
    logger.warning("pgvector not available, using JSON for embedding storage. Install with: pip install pgvector[sqlalchemy]")


class FaceEnrollment(Base):
    """Face enrollment with embeddings.
    
    Stores face embeddings for candidates. A candidate can have one enrollment
    that is used across all their exams (face doesn't change).
    """
    __tablename__ = "face_enrollments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(String(100), nullable=False, index=True)
    embedding_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Embedding stored in vector database (pgvector) for production-grade similarity search
    embedding = Column(JSON, nullable=False) # use json for dev for now
    embedding_dim = Column(Integer, default=512, nullable=False)
    
    # Quality metrics
    quality_score = Column(Float, nullable=False)
    sharpness_score = Column(Float, nullable=True)
    brightness_score = Column(Float, nullable=True)
    pose_angles = Column(JSON, nullable=True)  # {"pitch": float, "yaw": float, "roll": float}
    face_bbox = Column(JSON, nullable=True)  # [x1, y1, x2, y2]
    
    # Metadata
    source = Column(String(50), nullable=True)  # "web", "mobile", "kiosk"
    device_info = Column(JSON, nullable=True)
    enrollment_location = Column(String(200), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    extra_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    
    # Store the actual face crop used for enrollment (base64 encoded)
    face_image = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_enrollment_candidate", "candidate_id"),
        Index("idx_enrollment_quality", "quality_score"),
    )


class AlertLog(Base):
    """Ephemeral alert event logs."""
    __tablename__ = "alert_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(String(100), unique=True, nullable=False, index=True)
    exam_session_id = Column(String(200), nullable=False, index=True)
    candidate_id = Column(String(100), nullable=False, index=True)

    event_type = Column(String(100), nullable=False, index=True)
    severity_score = Column(Float, nullable=False)
    ai_model_source = Column(String(100), nullable=True)

    # Evidence and metadata
    evidence = Column(JSON, nullable=True)
    extra_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict

    # Dispatch status
    dispatched = Column(String(50), default="pending", nullable=False)  # "pending", "sent", "failed"
    dispatched_at = Column(DateTime(timezone=True), nullable=True)

    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_alert_session", "exam_session_id"),
        Index("idx_alert_candidate", "candidate_id"),
        Index("idx_alert_type", "event_type"),
        Index("idx_alert_timestamp", "timestamp"),
        Index("idx_alert_dispatched", "dispatched"),
    )