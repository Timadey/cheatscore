1. Comprehensive Model Pipeline (Backend)
1.1 Face Detection & Tracking Models
Primary: RetinaFace or SCRFD (Sample Cascade Relay Face Detection)

Purpose: Detect faces in each frame, handle multiple faces
Specs:

Input: 640×480 RGB frames
Output: Bounding boxes, 5 facial landmarks, confidence scores
Latency: ~10-15ms on GPU, ~50ms on CPU
Model size: ~2MB (SCRFD-0.5GN) to ~10MB (RetinaFace-R50)


Deployment: Run on every frame or every 3rd frame (10 FPS processing)
Alerts triggered:

MULTIPLE_FACES: >1 face detected for >3 consecutive seconds
FACE_NOT_FOUND: No face detected for >5 seconds
PRESENCE_LOSS: Face bounding box area <20% of enrollment baseline



Fallback: MediaPipe Face Detection (CPU-friendly)

Lightweight, optimized for mobile/web
Use when GPU unavailable

Implementation Note:
python# Detection frequency strategy
FRAME_SKIP = 2  # Process every 3rd frame (10 FPS from 30 FPS stream)
DETECTION_WINDOW = 3.0  # seconds for anomaly confirmation

1.2 Face Verification & Recognition Models
Primary: ArcFace (ResNet-50 or MobileFaceNet backbone)

Purpose: Generate 512-dim embeddings for enrollment and continuous verification
Specs:

Input: Aligned face crops (112×112)
Output: 512-dimensional L2-normalized embedding
Accuracy: 99.8% on LFW dataset
Latency: ~5ms (GPU) / ~30ms (CPU with MobileFaceNet)


Enrollment Process:

Capture 5-10 frames during enrollment
Select best 3 frames (highest quality score from face detector)
Generate embeddings, store average embedding vector
Store quality metrics: sharpness, illumination variance, pose angle


Continuous Verification:

Compare every 10th processed frame against enrolled embedding
Cosine similarity threshold: 0.75 (configurable per exam)
Alert on mismatch: similarity < threshold for >3 consecutive checks



Alternative: FaceNet or InsightFace

FaceNet: 128-dim embeddings (faster, slightly less accurate)
InsightFace: Modular, supports multiple backbones

Alert Schema:
json{
  "event_type": "FACE_MISMATCH",
  "severity_score": 0.92,
  "evidence": {
    "enrolled_embedding_id": "uuid",
    "current_similarity": 0.68,
    "threshold": 0.75,
    "frame_quality_score": 0.85,
    "consecutive_mismatches": 4
  }
}

1.3 Gaze Estimation & Eye Tracking
Primary: L2CS-Net (Gaze Estimation in the Wild)

Purpose: Detect if candidate is looking away from screen
Specs:

Input: Face crop (224×224) + facial landmarks
Output: Pitch & yaw angles (degrees)
Latency: ~8ms (GPU)


On-Screen Zone Definition:

python  # Calibration during first 30 seconds of exam
  PITCH_RANGE = (-15, 25)  # degrees, looking at screen
  YAW_RANGE = (-30, 30)    # degrees, centered

  # Alert if outside zone for >4 seconds

Alerts:

GAZE_OFF_SCREEN: Eyes outside acceptable range
PROLONGED_LOOK_AWAY: >10 seconds off-screen
FREQUENT_DISTRACTION: >5 off-screen events in 60 seconds



Alternative: MediaPipe Iris Tracking (lightweight)

Pupil tracking only, less accurate gaze vector
CPU-friendly for graceful degradation

Implementation Strategy:

Run gaze model every 5th processed frame (6 Hz from 30 FPS stream)
Use sliding window aggregation (3-second windows)
Suppress alerts during natural blinks (detected via eye aspect ratio)


1.4 Head Pose Estimation
Primary: 6D RepNet or FSA-Net

Purpose: Detect unusual head movements (looking down at phone, turning away)
Specs:

Input: Face crop with landmarks
Output: Roll, pitch, yaw (Euler angles)
Latency: ~6ms (GPU)


Baseline Pose: Establish during first 30 seconds

Pitch: -10° to +15° (natural screen viewing)
Yaw: -25° to +25°
Roll: -10° to +10°


Alerts:

HEAD_DOWN: Pitch < -30° for >3 seconds (looking at phone/notes)
HEAD_TURNED_AWAY: Yaw outside [-45°, 45°] for >3 seconds
SUSPICIOUS_MOVEMENT: Rapid head rotation (>60°/sec)



Integration with Gaze:

Combine head pose + gaze for higher confidence alerts
If head turned away AND gaze off-screen → higher severity


1.5 Body Pose & Movement Analysis
Primary: MediaPipe Pose (Holistic) or OpenPose

Purpose: Detect leaving seat, multiple people, unusual gestures
Specs:

Input: Full frame (640×480 or higher)
Output: 33 body keypoints (MediaPipe) or 18 (OpenPose)
Latency: ~20ms (GPU, MediaPipe), ~50ms (OpenPose)


Run Frequency: Every 10th frame (3 Hz) - less critical than face
Detections:

Presence: Upper body keypoints (shoulders, elbows) visible
Left Frame: Shoulders disappear for >5 seconds → LEFT_FRAME alert
Hand Movements: Track hand keypoints near face → HAND_NEAR_EAR (phone?)
Multiple Bodies: Detect >1 person skeleton → MULTIPLE_PERSONS



Alert Examples:
json{
  "event_type": "LEFT_FRAME",
  "severity_score": 0.88,
  "evidence": {
    "visible_keypoints": 5,  // out of 33
    "duration_seconds": 7.2,
    "last_seen_pose": "timestamp"
  }
}
Optimization:

Use lightweight MoveNet (Thunder or Lightning variant) for CPU deployments
Cache pose results across 3-5 frames for motion analysis


1.6 Audio Analysis Pipeline
A. Voice Activity Detection (VAD)

Model: Silero VAD or WebRTC VAD
Purpose: Detect speech activity → potential communication
Specs:

Input: 16kHz mono audio chunks (512ms)
Output: Speech probability (0-1)
Latency: <5ms (CPU-friendly)


Alert: VOICE_DETECTED if speech probability >0.8 for >2 seconds

B. Speaker Count Estimation

Model: pyannote.audio (speaker diarization)
Purpose: Detect multiple speakers → someone assisting
Specs:

Input: 5-second audio windows
Output: Number of unique speakers
Latency: ~200ms (GPU)


Alert: MULTIPLE_SPEAKERS if >1 speaker in 10-second window

C. Audio Anomaly Detection (Optional)

Model: AutoEncoder or YAMNet (audio classification)
Purpose: Detect keyboard typing, phone rings, environmental sounds
Specs:

YAMNet: Pretrained on AudioSet (521 classes)
Detect: "typing", "phone ringing", "door opening", "speech"


Alert: AUDIO_ANOMALY with detected class

D. Automatic Speech Recognition (ASR) - Optional

Model: Whisper (tiny/base) or Wav2Vec2
Purpose: Transcribe speech for keyword detection ("can you help me")
Privacy Consideration: Encrypt transcripts, delete after exam
Deployment: Only if explicitly enabled per exam policy

Audio Processing Strategy:
python# Audio chunk size: 512ms at 16kHz = 8192 samples
AUDIO_CHUNK_MS = 512
VAD_THRESHOLD = 0.7
DIARIZATION_WINDOW = 5  # seconds

# Run VAD on every chunk (CPU-friendly)
# Run diarization every 5 seconds if VAD positive
# Run ASR only if policy allows + VAD positive

1.7 Multi-Modal Anomaly Fusion (Meta-Model)
Purpose: Combine signals from all models for smarter alerting
Approach: Rule-based fusion with confidence scoring
python# Example: High-confidence cheating scenario
if (
    face_mismatch_count > 2 AND
    gaze_off_screen_duration > 10 AND
    voice_detected
):
    severity_score = 0.95
    event_type = "HIGH_CONFIDENCE_CHEATING"
Alternative: Train a lightweight ML classifier

Input: Feature vector [face_sim, gaze_angle, pose_delta, vad_score, ...]
Output: Cheating probability (0-1)
Dataset: Labeled exam videos (requires annotation)

Recommendation: Start with rule-based, collect data for future ML model

2. Frontend vs Backend Model Distribution
2.1 Recommended: Backend-Only (Primary Architecture)
Backend Processes:

All models listed above
Frame buffering & preprocessing
Alert generation & dispatch

Frontend Processes:

WebRTC streaming (capture + encode)
Display real-time feedback (alerts from backend via WebSocket)
Lightweight UI (timer, instructions)

Advantages:
✅ No model exposure (IP protection)
✅ Consistent detection across all devices
✅ GPU acceleration on backend
✅ Easy model updates (no frontend deployment)
✅ Lower frontend compute requirements
Disadvantages:
❌ Higher bandwidth usage (full video stream)
❌ Latency: ~200-500ms end-to-end (capture → inference → alert)
Bandwidth: ~500 kbps - 2 Mbps per stream (H.264, 720p @ 15-30 FPS)

2.2 Hybrid Alternative: Frontend Face Detection + Backend Verification
Frontend Processes:

Face Detection (MediaPipe or TensorFlow.js FaceDetection)

Detect face in frame
Crop face region
Send only face crop (not full frame)


Frame Selection: Send 1 frame per second instead of 30 FPS

Backend Processes:

Everything else (verification, gaze, pose, audio)

Advantages:
✅ 80% bandwidth reduction (send crops, not full frames)
✅ Reduced backend compute (no need to detect faces)
Disadvantages:
❌ Model exposed in frontend (can be extracted)
❌ Device compatibility issues (older phones, browsers)
❌ Inconsistent detection quality across devices
❌ Harder to debug (frontend + backend issues)
When to Use:

Limited backend GPU capacity
Network bandwidth constrained (rural areas, slow connections)
Mobile-first exam platform

Implementation Note:
javascript// Frontend: TensorFlow.js face detection
const model = await blazeface.load();
const predictions = await model.estimateFaces(video);

if (predictions.length > 0) {
  const faceCrop = cropFace(video, predictions[0]);
  sendToBackend(faceCrop);  // Send crop only
}
```

---

### 2.3 Not Recommended: Heavy Frontend Processing

**Why Avoid?**
- ❌ High battery drain on candidate devices
- ❌ Performance varies wildly (high-end vs low-end devices)
- ❌ Difficult to ensure model integrity (tampering)
- ❌ Privacy concerns (models running locally can be inspected)

**Exception**: Use for low-stakes quizzes or optional client-side warnings (not official alerts)

---

## 3. Refined System Architecture

### 3.1 Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (React/Flutter)                 │
│  ┌──────────────┐   ┌─────────────┐   ┌─────────────────┐  │
│  │ WebRTC       │   │ WebSocket   │   │ UI (Alerts,     │  │
│  │ Capture      │───│ Signaling   │───│ Feedback)       │  │
│  └──────────────┘   └─────────────┘   └─────────────────┘  │
└────────────┬────────────────────────────────────────────────┘
             │ Video/Audio Stream (H.264/Opus)
             ↓
┌─────────────────────────────────────────────────────────────┐
│              WEBRTC GATEWAY (Janus/Mediasoup)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • NAT Traversal (STUN/TURN)                         │   │
│  │  • Codec Negotiation                                 │   │
│  │  • Frame Extraction → gRPC/WebSocket                 │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────┬────────────────────────────────────────────────┘
             │ Frame Stream (gRPC Binary / WebSocket)
             ↓
┌─────────────────────────────────────────────────────────────┐
│           PROCTORING SERVICE (FastAPI + Workers)            │
│                                                              │
│  ┌──────────────────┐       ┌─────────────────────────┐    │
│  │  API Server      │       │  Session Manager        │    │
│  │  • /enroll       │◄──────┤  • Redis Session Cache  │    │
│  │  • /verify       │       │  • Candidate Mapping    │    │
│  │  • /session/*    │       └─────────────────────────┘    │
│  └──────────────────┘                                       │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         INFERENCE WORKER POOL (Multi-Process)         │  │
│  │                                                         │  │
│  │  ┌─────────────┐  ┌──────────┐  ┌─────────────────┐  │  │
│  │  │ Face        │  │ Gaze     │  │ Pose            │  │  │
│  │  │ Detection   │→ │ Tracking │→ │ Estimation      │  │  │
│  │  │ (RetinaFace)│  │(L2CS-Net)│  │ (MediaPipe)     │  │  │
│  │  └─────────────┘  └──────────┘  └─────────────────┘  │  │
│  │                                                         │  │
│  │  ┌─────────────┐  ┌──────────┐  ┌─────────────────┐  │  │
│  │  │ Face        │  │ Audio    │  │ Anomaly         │  │  │
│  │  │ Verification│  │ Analysis │  │ Fusion          │  │  │
│  │  │ (ArcFace)   │  │ (VAD+    │  │ (Rule Engine)   │  │  │
│  │  └─────────────┘  │Diarization│ └─────────────────┘  │  │
│  │                   └──────────┘                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓ Inference Events                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           ALERT DISPATCHER (Async Service)            │  │
│  │                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │  │
│  │  │ gRPC Stream  │  │ Kafka        │  │ Webhooks   │  │  │
│  │  │ (Primary)    │  │ (Durable)    │  │ (Fallback) │  │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬─────┘  │  │
│  │         └──────────────────┼──────────────────┘        │  │
│  └────────────────────────────┼───────────────────────────┘  │
│                                ↓                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  STORAGE                                              │   │
│  │  • Postgres (enrollment metadata)                    │   │
│  │  • Vector DB (embeddings: FAISS/Weaviate)            │   │
│  │  • Redis (session cache, frame buffers)              │   │
│  │  • S3 (optional: evidence frames/audio)              │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────────────────┘
                       │ Alerts (gRPC/Kafka/Webhook)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│            EXTERNAL DECISION BACKEND (Your System)          │
│  • Alert Storage (Postgres/MongoDB)                         │
│  • Business Logic (escalation, auto-fail, review queue)     │
│  • Admin Dashboard                                           │
│  • Notification Service (email, SMS to proctors)            │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.2 Data Flow: Frame Processing Pipeline
```
WebRTC Stream (30 FPS) → Janus Gateway
                            ↓
        ┌───────────────────┴────────────────────┐
        │ Frame Buffer (Redis/In-Memory)         │
        │ • Ring buffer: 5-second history        │
        │ • Indexed by exam_session_id           │
        └───────────┬────────────────────────────┘
                    ↓
        ┌───────────┴────────────────────────────┐
        │ Frame Sampler                          │
        │ • Every 3rd frame → Face models        │
        │ • Every 5th frame → Gaze model         │
        │ • Every 10th frame → Pose model        │
        └───────────┬────────────────────────────┘
                    ↓
        ┌───────────┴────────────────────────────┐
        │ Preprocessing                          │
        │ • Decode H.264 → RGB                   │
        │ • Resize (640x480)                     │
        │ • Normalize (0-1 or -1 to 1)           │
        └───────────┬────────────────────────────┘
                    ↓
        ┌───────────┴────────────────────────────┐
        │ Model Inference (Batch if GPU)         │
        │ • Batch size: 4-8 frames               │
        │ • Parallel execution per model         │
        └───────────┬────────────────────────────┘
                    ↓
        ┌───────────┴────────────────────────────┐
        │ Anomaly Detection Rules                │
        │ • Sliding window aggregation (3s)      │
        │ • Threshold checks                     │
        │ • Confidence scoring                   │
        └───────────┬────────────────────────────┘
                    ↓ (if anomaly detected)
        ┌───────────┴────────────────────────────┐
        │ Alert Generation                       │
        │ • Serialize to canonical JSON          │
        │ • Attach evidence (frame_ids, scores)  │
        │ • Add metadata (timestamps, model ver) │
        └───────────┬────────────────────────────┘
                    ↓
        ┌───────────┴────────────────────────────┐
        │ Alert Dispatcher                       │
        │ 1. gRPC stream (immediate)             │
        │ 2. Kafka publish (durable)             │
        │ 3. Webhook POST (fallback)             │
        └────────────────────────────────────────┘
```

---

## 4. Enhanced API Contract

### 4.1 Enrollment Endpoint (Enhanced)
```
POST /api/v1/enroll
Request:
json{
  "candidate_id": "CAND-2024-12345",
  "exam_id": "EXAM-CS101-FINAL",
  "images": [
    {
      "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
      "capture_timestamp": "2025-12-12T10:30:00Z",
      "device_info": {
        "camera_resolution": "1920x1080",
        "browser": "Chrome/120.0",
        "os": "Windows 10"
      }
    },
    // ... 3-5 images recommended
  ],
  "metadata": {
    "enrollment_location": "home",
    "ip_address": "192.168.1.100",
    "session_token": "eyJhbGc..."
  }
}
Response:
json{
  "status": "success",
  "candidate_id": "CAND-2024-12345",
  "enrollment_id": "ENR-uuid-12345",
  "embeddings": [
    {
      "embedding_id": "EMB-uuid-1",
      "quality_score": 0.94,
      "face_bbox": [120, 80, 350, 310],
      "pose_angles": {"pitch": 5, "yaw": -2, "roll": 1},
      "sharpness_score": 0.89,
      "selected": true  // Best embedding
    },
    // ... other embeddings
  ],
  "average_embedding_stored": true,
  "enrollment_timestamp": "2025-12-12T10:30:05Z",
  "recommendations": [
    "Good lighting detected",
    "Face centered and clear"
  ]
}
```

**Quality Checks During Enrollment**:
- Face size: >80×80 pixels
- Sharpness: Laplacian variance >100
- Illumination: Mean brightness 40-220
- Pose: Pitch, yaw, roll within ±20°
- Multiple faces: Reject if >1 face detected

---

### 4.2 Session Management (Enhanced)
```
POST /api/v1/session/start
Request:
json{
  "exam_session_id": "SESS-EXAM-CS101-CAND-12345",
  "candidate_id": "CAND-2024-12345",
  "exam_id": "EXAM-CS101-FINAL",
  "frontend_instance_id": "WEB-uuid-67890",
  "verification_policy": {
    "face_match_threshold": 0.75,
    "gaze_tolerance_degrees": 30,
    "max_offline_seconds": 10,
    "alert_sensitivity": "high"  // low, medium, high
  },
  "features_enabled": {
    "face_verification": true,
    "gaze_tracking": true,
    "pose_analysis": true,
    "audio_monitoring": true,
    "speech_transcription": false  // Privacy opt-in
  },
  "metadata": {
    "exam_duration_minutes": 120,
    "start_time": "2025-12-12T14:00:00Z"
  }
}
Response:
json{
  "status": "active",
  "session_id": "SESS-EXAM-CS101-CAND-12345",
  "websocket_url": "wss://proctoring.example.com/ws/SESS-...",
  "webrtc_config": {
    "ice_servers": [
      {"urls": "stun:stun.example.com:3478"},
      {
        "urls": "turn:turn.example.com:3478",
        "username": "user123",
        "credential": "pass456"
      }
    ],
    "video_constraints": {
      "width": {"ideal": 1280},
      "height": {"ideal": 720},
      "frameRate": {"ideal": 30, "max": 30}
    },
    "audio_constraints": {
      "sampleRate": 16000,
      "channelCount": 1,
      "echoCancellation": true
    }
  },
  "calibration_required": true,
  "calibration_duration_seconds": 30,
  "session_expires_at": "2025-12-12T16:00:00Z"
}
```

---

### 4.3 Real-Time Verification Endpoint
```
POST /api/v1/session/{session_id}/verify-frame
Use Case: Ad-hoc verification during session (triggered by frontend or proctor)
Request:
json{
  "frame_id": "FRAME-uuid-12345",  // References buffered frame
  "timestamp": "2025-12-12T14:15:30Z",
  "trigger": "manual_proctor_check"
}
Response:
json{
  "match": true,
  "similarity_score": 0.87,
  "threshold_used": 0.75,
  "comparison_embedding_id": "EMB-uuid-1",
  "face_quality": {
    "sharpness": 0.82,
    "pose_deviation": 8.5,  // degrees from frontal
    "lighting_quality": 0.91
  },
  "timestamp": "2025-12-12T14:15:31Z"
}

4.4 WebSocket Real-Time Alerts
Connection: wss://proctoring.example.com/ws/{session_id}
Alert Message Format (sent to frontend):
json{
  "type": "alert",
  "event_id": "EVT-uuid-12345",
  "exam_session_id": "SESS-EXAM-CS101-CAND-12345",
  "event_type": "GAZE_OFF_SCREEN",
  "severity_score": 0.72,
  "timestamp": "2025-12-12T14:20:15Z",
  "message": "Candidate looking away from screen",
  "action_required": "warning",  // warning, escalate, auto-pause
  "evidence_thumbnail": "data:image/jpeg;base64,/9j/4AAQ...",
  "dismiss_after_seconds": 10
}
Heartbeat (every 10 seconds):
json{
  "type": "heartbeat",
  "session_id": "SESS-...",
  "status": "active",
  "stats": {
    "frames_processed": 18450,
    "alerts_triggered": 3,
    "last_face_verification": "2025-12-12T14:20:10Z"
  }
}

5. Model Deployment Strategy
5.1 GPU-Accelerated Deployment (Recommended)
Infrastructure:

Kubernetes cluster with GPU node pools (NVIDIA T4, A10, or A100)
Triton Inference Server for model serving
HPA (Horizontal Pod Autoscaler) based on GPU utilization

Model Server Configuration:
yaml# Triton model repository structure
models/
├── face_detection/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt  # batch_size: 8, instance_count: 2
├── face_verification/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt  # batch_size: 16, instance_count: 1
├── gaze_estimation/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
└── pose_estimation/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt
Worker Architecture:
python# inference_worker.py
import tritonclient.grpc as grpcclient
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class InferenceWorker:
    def __init__(self, triton_url="localhost:8001"):
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def process_frame(self, frame: np.ndarray, session_id: str):
        # Parallel model execution
        futures = []

        # Face detection (required first)
        faces = await self.detect_faces(frame)

        if len(faces) == 0:
            return {"alert": "FACE_NOT_FOUND"}

        # Parallel execution of dependent models
        futures.append(self.verify_face(faces[0], session_id))
        futures.append(self.estimate_gaze(faces[0]))
        futures.append(self.estimate_pose(faces[0]))

        results = await asyncio.gather(*futures)

        # Anomaly fusion
        alerts = self.fuse_results(results, session_id)
        return alerts

    async def detect_faces(self, frame):
        # Triton inference call
        inputs = [grpcclient.InferInput("input", frame.shape, "FP32")]
        inputs[0].set_data_from_numpy(frame)

        response = await self.triton_client.async_infer(
            model_name="face_detection",
            inputs=inputs
        )
        return self.parse_detections(response)
Batching Strategy:

Dynamic batching in Triton (max_batch_size=8, max_queue_delay_microseconds=5000)
Accumulate frames from multiple sessions for efficient GPU utilization


5.2 CPU-Only Deployment (Continue5:12 PMGraceful Degradation)
Lightweight Models:

Face Detection: SCRFD-0.5GN (2MB, ONNX)
Face Verification: MobileFaceNet (4MB)
Gaze: Simplified L2CS (no ResNet backbone)
Pose: MoveNet Lightning (5MB)
Audio: Silero VAD only (no diarization)

Performance:

Latency: ~150-200ms per frame (vs ~30ms on GPU)
Reduce frame processing rate: 5 FPS instead of 10 FPS
Disable pose estimation if CPU usage >80%

Auto-Scaling Logic:
python# k8s HPA with custom metrics
if gpu_available:
    target_fps = 10
    models = ["face_detection", "face_verification", "gaze", "pose", "audio"]
else:
    target_fps = 5
    models = ["face_detection", "face_verification", "gaze_lite", "audio_vad"]
```

---

### 5.3 Model Version Management

**Versioning Strategy**:
```
models/
├── face_verification/
│   ├── 1/  # v1.0 (ArcFace ResNet-50)
│   ├── 2/  # v1.1 (ArcFace MobileFaceNet - faster)
│   └── config.pbtxt  # Default version: 2
A/B Testing:

Route 10% of traffic to new model version
Compare alert rates, false positive rates
Gradual rollout if metrics improve

Rollback Plan:

Keep last 2 versions deployed
Feature flag per exam: "model_version": "1" to force specific version


6. Alert Dispatcher Implementation (Code Scaffold)Alert Dispatcher ServiceCode # alert_dispatcher.py
"""
Production-grade Alert Dispatcher for AI Proctoring System
Supports gRPC streaming (primary), Kafka (durable), and Webhooks (fallback)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict

7. Security & Privacy Considerations
7.1 Data Protection
Encryption at Rest:

Face embeddings: Encrypt in database (AES-256)
Frame buffers: Optional encryption (performance trade-off)
Audit logs: Encrypt sensitive fields

Encryption in Transit:

WebRTC: DTLS-SRTP (mandatory)
gRPC: TLS 1.3 with mTLS for alert dispatcher
API endpoints: HTTPS only

Data Retention Policy:
- Face embeddings: Delete 90 days after exam
- Frame buffers: Ephemeral (5-second ring buffer, no persistence)
- Evidence frames: Delete 30 days after exam (or per policy)
- Audit logs: 1 year retention, then archive
- Alerts: Ownership transfers to external backend

7.2 Authentication & Authorization
API Authentication:

JWT tokens for API access
Service-to-service: mTLS certificates
Candidate authentication: OAuth2 (delegated to external system)

RBAC (Role-Based Access Control):
Roles:
- candidate: /enroll, /session/start (self only)
- proctor: /session/*, /verify-frame
- admin: /admin/*, access to all sessions
- system: Internal service calls

7.3 Privacy Compliance (GDPR, CCPA)
Consent Management:

Explicit consent before enrollment: "I consent to AI proctoring..."
Checkbox for optional features (speech transcription)

Data Subject Rights:

/api/v1/privacy/export - Export candidate's data (GDPR Article 20)
/api/v1/privacy/delete - Delete candidate's data (Right to erasure)

Transparency:

Provide documentation: "How AI Proctoring Works"
Disclose models used, accuracy rates, false positive rates


8. Monitoring & Observability
8.1 Prometheus Metrics (Key Indicators)
python# metrics.py (using prometheus_client)
from prometheus_client import Counter, Histogram, Gauge

# Inference metrics
frames_processed = Counter('frames_processed_total', 'Total frames processed', ['session_id', 'model'])
inference_latency = Histogram('inference_latency_seconds', 'Model inference latency', ['model'])
model_errors = Counter('model_errors_total', 'Model inference errors', ['model', 'error_type'])

# Alert metrics
alerts_generated = Counter('alerts_generated_total', 'Alerts generated', ['event_type', 'severity'])
alert_dispatch_latency = Histogram('alert_dispatch_latency_seconds', 'Alert dispatch latency', ['transport'])

# Session metrics
active_sessions = Gauge('active_exam_sessions', 'Currently active exam sessions')
webrtc_connection_errors = Counter('webrtc_connection_errors_total', 'WebRTC connection errors', ['error_type'])

# Resource metrics
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
inference_queue_depth = Gauge('inference_queue_depth', 'Frames waiting for inference')
```

---

### 8.2 Distributed Tracing (OpenTelemetry)

**Trace Spans**:
```
Trace: Exam Session (2 hours)
├─ Span: Frame Ingestion (continuous)
│  ├─ Span: Face Detection (10ms)
│  ├─ Span: Face Verification (5ms)
│  ├─ Span: Gaze Estimation (8ms)
│  └─ Span: Pose Estimation (12ms)
├─ Span: Alert Generation (1ms)
└─ Span: Alert Dispatch (20ms)
   ├─ Span: gRPC Send (5ms)
   └─ Span: Kafka Publish (15ms)
Tracing Context Propagation:

Inject trace_id in WebRTC metadata
Propagate through inference pipeline
Include in alert payloads for end-to-end tracking


8.3 Logging Strategy
Structured Logs (JSON):
json{
  "timestamp": "2025-12-12T14:20:15.123Z",
  "level": "WARNING",
  "service": "proctoring-worker-3",
  "trace_id": "abc123",
  "exam_session_id": "SESS-...",
  "candidate_id": "CAND-12345",
  "event": "face_mismatch_detected",
  "similarity_score": 0.68,
  "threshold": 0.75,
  "consecutive_failures": 3,
  "model_version": "arcface-v1.3"
}
Log Levels:

DEBUG: Frame-level processing (disabled in production)
INFO: Session start/end, model loading
WARNING: Detection anomalies, retry attempts
ERROR: Model failures, dispatch failures
CRITICAL: Service crashes, data corruption


9. Deployment & Scaling
9.1 Kubernetes Deployment (Helm Chart)
yaml# values.yaml (simplified)
replicaCount: 3  # API server replicas

inference:
  workers: 10  # Inference worker pods
  gpu:
    enabled: true
    type: nvidia-tesla-t4
    count: 1  # GPUs per worker
  resources:
    requests:
      cpu: "4"
      memory: "8Gi"
    limits:
      cpu: "8"
      memory: "16Gi"

alertDispatcher:
  replicaCount: 2
  kafka:
    bootstrapServers: "kafka-cluster:9092"
  grpc:
    endpoint: "external-backend.example.com:50051"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetGPUUtilization: 70
  targetCPUUtilization: 75

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
  jaeger:
    enabled: true

9.2 Scaling Strategy
Horizontal Scaling Triggers:

Active sessions > 100: Scale inference workers (10 workers per 50 sessions)
GPU utilization > 80%: Add GPU nodes
Alert dispatch queue depth > 1000: Scale dispatcher replicas

Vertical Scaling:

Increase worker memory if OOM errors
Increase GPU memory for larger batch sizes

Load Balancing:

Session-affinity load balancing (sticky sessions by exam_session_id)
Round-robin for stateless API calls (/enroll, /verify)


9.3 Cost Optimization
GPU Cost Reduction:

Use spot instances for non-critical inference (dev/test environments)
Batch multiple sessions per GPU (up to 4 concurrent sessions per T4 GPU)
Auto-scale down during off-peak hours (nights, weekends)

Network Cost:

Use regional WebRTC gateways (reduce inter-region bandwidth)
Compress video streams (H.264 with CRF 28-30)
Store evidence frames in regional S3 buckets

Estimated Cost (AWS):

1 GPU worker (g4dn.xlarge): ~$0.50/hour
100 concurrent sessions: ~10 GPU workers = $5/hour = $120/day
Kafka managed (MSK): ~$250/month
S3 storage (evidence frames): ~$0.03/GB/month


10. Testing & Validation Strategy
10.1 Unit Tests
python# test_face_verification.py
import pytest
from inference.face_verification import FaceVerifier

def test_face_verification_match():
    verifier = FaceVerifier()
    embedding1 = verifier.extract_embedding(load_test_image("candidate_enroll.jpg"))
    embedding2 = verifier.extract_embedding(load_test_image("candidate_exam.jpg"))

    similarity = verifier.compute_similarity(embedding1, embedding2)
    assert similarity > 0.75, "Same person should have high similarity"

def test_face_verification_mismatch():
    verifier = FaceVerifier()
    embedding1 = verifier.extract_embedding(load_test_image("candidate1.jpg"))
    embedding2 = verifier.extract_embedding(load_test_image("candidate2.jpg"))

    similarity = verifier.compute_similarity(embedding1, embedding2)
    assert similarity < 0.60, "Different people should have low similarity"

10.2 Integration Tests
python# test_alert_dispatcher.py
@pytest.mark.asyncio
async def test_alert_dispatch_kafka():
    dispatcher = AlertDispatcher(
        kafka_bootstrap_servers=["localhost:9092"],
        enable_grpc=False,
        enable_webhook=False
    )
    await dispatcher.initialize()

    alert = AlertEvent(
        event_id="TEST-001",
        exam_session_id="SESS-TEST",
        candidate_id="CAND-TEST",
        timestamp="2025-12-12T10:00:00Z",
        event_type="FACE_MISMATCH",
        severity_score=0.85,
        ai_model_source="test",
        evidence={},
        metadata={}
    )

    result = await dispatcher.dispatch(alert)
    assert result["kafka"] == True, "Kafka dispatch should succeed"

    await dispatcher.close()

10.3 End-to-End Tests
Scenario: Full Exam Session

Enroll candidate with 5 images
Start exam session via API
Stream simulated WebRTC video (30 FPS, 10 minutes)
Inject cheating scenarios:

Face swap at 2:00 mark
Look away at 5:00 mark
Leave frame at 7:30 mark


Verify alerts generated correctly
Verify alerts received by external backend
End session and verify cleanup

Load Testing:

Tool: Locust or k6
Simulate 500 concurrent exam sessions
Measure: P50, P95, P99 latency, error rates, resource usage


11. Disaster Recovery & High Availability
11.1 Backup Strategy
Databases:

Postgres (enrollment data): Daily backups, 30-day retention
Vector DB (embeddings): Incremental backups every 6 hours

Kafka:

Replication factor: 3 (data survives 2 broker failures)
Cross-region replication for DR

Redis:

AOF (Append-Only File) enabled for session state
Snapshot every 5 minutes


11.2 Failover Strategy
Multi-Region Deployment:

Primary region: us-east-1
Secondary region: eu-west-1
Failover: DNS-based (Route53 health checks)

Service Dependencies:

If Kafka unavailable: Buffer alerts in Redis (up to 100K alerts)
If GPU workers unavailable: Fallback to CPU workers (degraded mode)
If external backend unavailable: Continue generating alerts (buffered in Kafka)


Summary & Next Steps
Recommended Model Pipeline (Backend)

Face Detection: RetinaFace (GPU) / SCRFD (CPU)
Face Verification: ArcFace MobileFaceNet
Gaze Tracking: L2CS-Net
Head Pose: 6D RepNet or FSA-Net
Body Pose: MediaPipe Pose (or skip if CPU-limited)
Audio VAD: Silero VAD
Speaker Diarization: pyannote.audio (optional)

Deployment Priority
Phase 1 (MVP):

Face enrollment + verification
Basic gaze tracking
Alert dispatcher (Kafka only)
50 concurrent sessions

Phase 2 (Production):

Full model pipeline
gRPC alert streaming
GPU acceleration
500 concurrent sessions

Phase 3 (Scale):

Multi-region deployment
Advanced anomaly fusion
ASR transcription (optional)
5,000+ concurrent sessions