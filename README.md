# SD Proctor - AI-Powered Proctoring Service

An AI-powered proctoring backend service built with Python and FastAPI. Connects to frontend applications via WebRTC to receive audio and video data for real-time cheating detection during examinations.

## Features

- **Face Enrollment & Verification**: Enroll candidates and continuously verify identity during exams
- **Real-time Cheating Detection**:
  - Face verification and mismatch detection
  - Gaze tracking (eye off-screen detection)
  - Head pose estimation
  - Body movement analysis
  - Audio anomaly detection (voice activity, multiple speakers)
  - Multiple person detection
- **WebRTC Integration**: Direct audio/video streaming from frontend applications
- **Multi-Transport Alert Dispatch**: gRPC, Kafka, and Webhooks
- **Scalable Architecture**: GPU-accelerated inference with graceful CPU fallback

## Architecture

```
Frontend (WebRTC) → Gateway Adapter → Inference Workers → Alert Dispatcher → External Backend
                                            ↓
                                    Face/Gaze/Pose/Audio Models
                                            ↓
                                    PostgreSQL + Redis + Vector DB
```

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL
- Redis
- Kafka (optional, for durable alerts)
- CUDA-capable GPU (optional, for acceleration)

### Installation

```bash
# Clone the repository
cd /Users/macbook/PycharmProjects/sd-proctor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
alembic upgrade head

# Start the service
python -m app.main
```

### Development Mode

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Enrollment
- `POST /api/v1/enroll` - Enroll candidate face
- `GET /api/v1/enroll/{candidate_id}` - Get enrollment details
- `DELETE /api/v1/enroll/{candidate_id}` - Delete enrollment

### Verification
- `POST /api/v1/verify` - Ad-hoc face verification

### Session Management
- `POST /api/v1/session/start` - Start exam session
- `POST /api/v1/session/{session_id}/end` - End exam session
- `GET /api/v1/session/{session_id}` - Get session details
- `WS /api/v1/session/ws/{session_id}` - WebSocket for real-time alerts

### Admin
- `GET /api/v1/admin/stats` - System statistics
- `GET /api/v1/admin/sessions` - List active sessions

### Health & Metrics
- `GET /.well-known/health` - Health check
- `GET /metrics` - Prometheus metrics

## Project Structure

```
sd-proctor/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   ├── models.py               # Database models
│   ├── schemas.py              # Pydantic schemas
│   ├── api/
│   │   └── v1/
│   │       ├── enrollment.py   # Enrollment endpoints
│   │       ├── verification.py # Verification endpoints
│   │       ├── session.py      # Session endpoints
│   │       └── admin.py        # Admin endpoints
│   ├── inference/
│   │   ├── face_detection.py   # Face detection models
│   │   ├── face_verification.py # Face verification models
│   │   ├── gaze_estimation.py  # Gaze tracking models
│   │   ├── pose_estimation.py  # Pose detection models
│   │   └── audio_analysis.py   # Audio processing models
│   ├── dispatcher/
│   │   └── alert_dispatcher.py # Alert dispatch system
│   ├── webrtc/
│   │   └── gateway_adapter.py  # WebRTC gateway adapter
│   ├── services/
│   │   ├── enrollment_service.py
│   │   ├── verification_service.py
│   │   └── session_service.py
│   └── utils/
│       ├── db.py               # Database utilities
│       ├── redis_client.py     # Redis client
│       └── metrics.py          # Prometheus metrics
├── alembic/                    # Database migrations
├── tests/                      # Test suite
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## Configuration

Key configuration parameters in `.env`:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka brokers
- `MODEL_DEVICE`: `cuda` or `cpu`
- `FACE_MATCH_THRESHOLD`: Face verification threshold (default: 0.75)
- `TRITON_ENABLED`: Enable Triton Inference Server

## Model Pipeline

### Phase 1 (MVP)
1. Face Detection (RetinaFace/SCRFD)
2. Face Verification (ArcFace)
3. Basic Alert Dispatcher (Kafka)

### Phase 2 (Production)
4. Gaze Tracking (L2CS-Net)
5. Head Pose Estimation (6D RepNet)
6. Audio VAD (Silero VAD)
7. gRPC Alert Streaming

### Phase 3 (Advanced)
8. Body Pose (MediaPipe Pose)
9. Speaker Diarization (pyannote.audio)
10. ASR Transcription (Whisper)

## Alert Event Schema

```json
{
  "event_id": "uuid",
  "exam_session_id": "string",
  "candidate_id": "string",
  "timestamp": "ISO8601",
  "event_type": "FACE_MISMATCH | GAZE_OFF_SCREEN | ...",
  "severity_score": 0.85,
  "ai_model_source": "arcface-v1.3",
  "evidence": {
    "frame_ids": ["..."],
    "confidence_metrics": {"face_sim": 0.68}
  },
  "metadata": {}
}
```

## Deployment

### Docker
```bash
docker build -t sd-proctor:latest .
docker run -p 8000:8000 --env-file .env sd-proctor:latest
```

### Kubernetes
```bash
helm install sd-proctor ./helm/sd-proctor
```

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

## Monitoring

- Prometheus metrics: `http://localhost:9090/metrics`
- Health check: `http://localhost:8000/.well-known/health`

## Security & Privacy

- All data encrypted in transit (TLS/DTLS-SRTP)
- Face embeddings encrypted at rest (AES-256)
- GDPR-compliant data deletion endpoints
- Configurable data retention policies

## License

[Your License Here]

## Next Steps

See ROADMAP.md for implementation priorities.
