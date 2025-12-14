# Frontend Test Guide

## Overview

This guide explains how to test the SD Proctor service using the provided frontend test page.

## Prerequisites

1. **Backend Service Running**:
   ```bash
   # Start the backend service
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Services Running**:
   - PostgreSQL (for enrollment storage)
   - Redis (for session state and frame buffer)
   - Kafka (for alert dispatch) - optional for basic testing

3. **Browser Requirements**:
   - Modern browser with WebRTC support (Chrome, Firefox, Edge)
   - HTTPS or localhost (required for camera access)

## Quick Start

1. **Open the Test Page**:
   - Open `frontend_test.html` in your browser
   - Or serve it via a simple HTTP server:
     ```bash
     python -m http.server 8080
     ```
     Then navigate to `http://localhost:8080/frontend_test.html`

2. **Test Flow**:
   - **Step 1**: Enroll a candidate's face
   - **Step 2**: Start an exam session
   - **Step 3**: Start video stream for real-time monitoring

## Testing Steps

### 1. Face Enrollment

1. Enter a Candidate ID (e.g., "CAND-001")
2. Click "Capture & Enroll Face"
3. Allow camera access when prompted
4. The system will capture 5 images automatically
5. Wait for enrollment confirmation

**Expected Result**: 
- Success message with quality score
- Enrollment stored in database

### 2. Start Exam Session

1. Enter Exam Session ID (e.g., "SESS-001")
2. Enter Candidate ID (must match enrolled candidate)
3. Click "Start Session"

**Expected Result**:
- Session started message
- WebSocket connection established for alerts
- Connection status shows "Connected"

### 3. Real-Time Video Stream

1. Click "Start Video Stream"
2. Camera will start capturing
3. Frames are sent to backend at 30 FPS
4. Alerts will appear in real-time if anomalies detected

**Expected Result**:
- Video preview visible
- Frame count increasing
- Alerts appear if face mismatch detected

## Testing Scenarios

### Scenario 1: Normal Face Match
- Enroll your face
- Start session
- Keep your face visible and centered
- **Expected**: No alerts, frame count increases

### Scenario 2: Face Mismatch
- Enroll your face
- Start session
- Have someone else sit in front of camera
- **Expected**: FACE_MISMATCH alert after 3 consecutive mismatches

### Scenario 3: Multiple Faces
- Enroll your face
- Start session
- Have multiple people visible in camera
- **Expected**: MULTIPLE_FACES alert after 3 seconds

### Scenario 4: Face Not Found
- Enroll your face
- Start session
- Move away from camera or cover face
- **Expected**: FACE_NOT_FOUND alert after 5 seconds

## API Endpoints Used

### Enrollment
- `POST /api/v1/enroll` - Enroll candidate face

### Session Management
- `POST /api/v1/session/start` - Start exam session
- `WS /api/v1/session/ws/{session_id}` - WebSocket for alerts

### WebRTC Frame Ingestion
- `WS /api/v1/webrtc/frames/{session_id}` - WebSocket for frame upload

## WebSocket Message Formats

### Frame Upload (Client → Server)
```json
{
    "type": "frame",
    "candidate_id": "CAND-001",
    "frame_data": "base64_encoded_jpeg",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

### Alert (Server → Client)
```json
{
    "type": "alert",
    "event_id": "uuid",
    "event_type": "FACE_MISMATCH",
    "severity_score": 0.95,
    "timestamp": "2024-01-01T00:00:00Z",
    "message": "Alert: FACE_MISMATCH"
}
```

### Heartbeat (Server → Client)
```json
{
    "type": "heartbeat",
    "session_id": "SESS-001",
    "status": "active",
    "stats": {
        "frame_count": 150,
        "verification_count": 15
    }
}
```

## Troubleshooting

### Camera Not Accessing
- Ensure you're using HTTPS or localhost
- Check browser permissions
- Try a different browser

### WebSocket Connection Failed
- Check backend is running on port 8000
- Check CORS settings
- Check browser console for errors

### No Alerts Generated
- Verify candidate is enrolled
- Check face is visible in camera
- Verify session is active
- Check backend logs for errors

### Frames Not Sending
- Check WebSocket connection status
- Verify session is started
- Check browser console for errors
- Ensure camera is working

## Performance Notes

- **Frame Rate**: 30 FPS (configurable)
- **Frame Size**: 640x480 (configurable)
- **Compression**: JPEG quality 0.8
- **Latency**: <500ms from frame capture to alert

## Next Steps

After testing with the frontend:
1. Integrate with your actual frontend application
2. Customize alert handling
3. Add additional detection models (gaze, pose, audio)
4. Scale for production deployment

## Support

For issues or questions:
- Check backend logs: `tail -f logs/app.log`
- Check browser console for errors
- Verify all services are running
- Review API documentation at `/docs`

