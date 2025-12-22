SD Proctor — WebSocket Usage
=============================

Purpose
-------
Real-time transport for extracted frame features from the frontend to the backend.
Messages are stored as short-lived metadata frames in Redis and used for live analysis and final session analysis.

Endpoint
--------
- ws://{HOST}/api/v1/session/ws/{session_id}
  - Example: ws://localhost:8000/api/v1/session/ws/demo_session

Connection lifecycle
--------------------
- Client connects with the exam session_id returned from the Start API.
- Server starts two loops per connection:
  - receive_loop: accepts incoming JSON payloads and stores them in Redis (metadata frames).
  - heartbeat_loop: sends a heartbeat every 10s:
    { "type": "heartbeat", "session_id": "<id>", "timestamp": "<iso>" }
- When session ends the server sends:
  { "type": "session_ended", "session_id": "<id>", "timestamp": "<iso>" }
  and the receive loop stops.

Client → Server message formats
-------------------------------
Server accepts either a wrapper object or raw feature JSON. Recommended formats:

1) Wrapper (recommended), use @timadey/proctor to get extracted featured onEvent of engine.
```
{
  "extracted_features": { /* feature fields */ }
}
```

2) Typed wrapper
```
{
  "type": "frame",
  "payload": { /* feature fields */ }
}
```
3) Raw feature dict
```
{
  "session_id": "demo_session", // required, don't forget!
  "face_present": 1,
  "no_of_face": 1,
  "face_x": 260,
  "face_y": 312,
  "face_w": 150,
  "face_h": 150,
  "face_conf": 88,
  "head_yaw": 0.01,
  "head_pitch": 0.018,
  "head_roll": -0.0005,
  "head_pose": "forward",
  "gaze_on_script": 0,
  "gaze_direction": "bottom_right",
  "gazePoint_x": 500,
  "gazePoint_y": 400,
  "phone_present": 1,
  "phone_conf": 87,
  "hand_count": 1
}
```

Notes
-----
- Messages are stored under Redis keys frames_meta:{session_id} and frame_meta:{session_id}:{frame_id}.
- Retention is controlled by settings.frame_buffer_retention_seconds (default short). Increase if you need longer in-memory history.
- The server stores metadata only; images are not transmitted via this websocket in the current design.

Server endpoints related to analysis and LiveKit
-----------------------------------------------
- POST /api/v1/session/{session_id}/analyze
  - Runs on-demand analysis using frames in Redis (if session active) or returns persisted analysis (if ended and stored in DB).
  - Throttled to once per second per session (returns 429 if exceeded).

- POST /api/v1/session/{session_id}/end
  - Marks session ended, runs final analysis, persists report into DB (exam_analyses), dumps metadata frames to disk and clears Redis, then notifies connected websocket clients with session_ended. Returns 204.

- GET /api/v1/webrtc/livekit/token?exam_id=...&member_id=...&name=...
  - Returns a LiveKit access token for joining a room. Requires LiveKit credentials configured in settings.

Quick curl examples
-------------------
# Request on-demand analysis (throttled 1 req/sec)
```
curl -X POST "http://localhost:8000/api/v1/session/demo_session/analyze" -H "Accept: application/json"
```
# End session (triggers final analysis)
```
curl -X POST "http://localhost:8000/api/v1/session/demo_session/end" -i
```
# Get LiveKit token
```
curl -G "http://localhost:8000/api/v1/webrtc/livekit/token" \
  --data-urlencode "exam_id=exam123" \
  --data-urlencode "member_id=user123" \
  --data-urlencode "name=Candidate+One"
```
Browser JavaScript examples
---------------------------
```
// WebSocket connect and send feature message
const ws = new WebSocket("ws://localhost:8000/api/v1/session/ws/demo_session");
ws.onopen = () => {
  const payload = {
    frame_data: {
      session_id: "demo_session",
      face_present: 1,
      no_of_face: 1,
      face_x: 260,
      face_y: 312,
      phone_present: 0
    }
  };
  ws.send(JSON.stringify(payload));
};

ws.onmessage = (evt) => {
  const msg = JSON.parse(evt.data);
  console.log("WS message:", msg);
  if (msg.type === "session_ended") {
    console.log("Session ended by server");
    ws.close();
  }
};

// On-demand analysis (fetch)
fetch(`/api/v1/session/demo_session/analyze`, { method: 'POST' })
  .then(res => res.json())
  .then(report => console.log('Analysis report', report))
  .catch(console.error);

// End session
fetch(`/api/v1/session/demo_session/end`, { method: 'POST' })
  .then(res => { if (res.status === 204) console.log('Ended'); })
  .catch(console.error);
```
Behavior notes
--------------
- Because frame retention is short, call the analyze endpoint while the session is active to analyze live frames.
- Final analyses are persisted in the DB (exam_analyses table) and frames are dumped to analysis_dumps/{session_id}_frames.json in repo root.

Contact / Troubleshooting
-------------------------
- Ensure Redis and database connections are configured in .env (settings.redis_url, settings.database_url).
- If you see 429 on analyze requests, wait 1s before retrying.

EOF

