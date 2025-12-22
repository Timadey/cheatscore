"""
Real-Time Cheating Detection Inference
=======================================

This script demonstrates how to use trained models for real-time
cheating detection during live exam proctoring.

Features:
- Load pre-trained models
- Process incoming frames in real-time
- Buffering and smoothing for stable predictions
- Alert generation with different severity levels
- Integration-ready code for your proctoring system
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import json
from datetime import datetime
from collections import deque

from app.prediction.detectors import XGBoostCheatingDetector, LSTMCheatingDetector, RealTimeDetector
from app.prediction.utils import FeatureEngineer


# from train import (
#     XGBoostCheatingDetector,
#     LSTMCheatingDetector,
#     FeatureEngineer,
#     RealTimeDetector
# )


class LiveProctoringMonitor:
    """
    Real-time proctoring monitor that integrates with your existing system.
    """

    def __init__(self, model_path: str, model_type: str = 'xgboost',
                 buffer_size: int = 30, confidence_threshold: float = 0.7):
        """
        Initialize the monitor.

        Args:
            model_path: Path to saved model
            model_type: 'xgboost' or 'lstm'
            buffer_size: Number of frames to buffer
            confidence_threshold: Threshold for alerting
        """
        self.model_type = model_type
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold

        # Load model
        print(f"Loading {model_type} model from {model_path}...")
        if model_type == 'xgboost':
            self.model = XGBoostCheatingDetector()
            self.model.load(model_path)
        elif model_type == 'lstm':
            self.model = LSTMCheatingDetector()
            self.model.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(window_size=buffer_size)

        # Initialize real-time detector
        self.detector = RealTimeDetector(
            model=self.model,
            buffer_size=buffer_size,
            smoothing_window=5,
            confidence_threshold=confidence_threshold
        )

        # Session state
        self.current_session_id = None
        self.frame_count = 0
        self.alerts_history = []
        self.suspicious_events_log = []

        print(f"Monitor initialized successfully!")
        print(f"  Model: {model_type}")
        print(f"  Buffer size: {buffer_size} frames")
        print(f"  Confidence threshold: {confidence_threshold}")

    def process_frame(self, frame_data: Dict) -> Dict:
        """
        Process a single frame from your MediaPipe + browser telemetry.

        Args:
            frame_data: Dictionary containing frame features from your system
                Expected keys (from your MediaPipe output):
                - face_present, no_of_face, face_x, face_y, face_w, face_h
                - head_pose, head_pitch, head_yaw, head_roll
                - gaze_on_script, gaze_direction, gazePoint_x, gazePoint_y
                - phone_present, phone_conf
                - hand_count
                - etc.

        Returns:
            Dictionary with detection results and alerts
        """
        self.frame_count += 1

        # Add session metadata
        if self.current_session_id is None:
            self.current_session_id = frame_data.get('session_id', 'default_session')

        frame_data['session_id'] = self.current_session_id
        frame_data['row_id'] = self.frame_count

        # Add derived features on-the-fly
        frame_data = self._add_derived_features(frame_data)

        # Log suspicious events
        self._log_suspicious_events(frame_data)

        # Get detection result
        result = self.detector.add_frame(frame_data)

        # If buffer not full yet, return empty result
        if result is None:
            return {
                'status': 'buffering',
                'frames_buffered': len(self.detector.frame_buffer),
                'buffer_size': self.buffer_size,
                'message': f'Collecting frames... ({len(self.detector.frame_buffer)}/{self.buffer_size})'
            }

        # Process detection result
        detection_result = self._process_detection_result(result, frame_data)

        return detection_result

    def _add_derived_features(self, frame_data: Dict) -> Dict:
        """Add derived features to frame data."""
        # Face area
        if 'face_w' in frame_data and 'face_h' in frame_data:
            frame_data['face_area'] = frame_data['face_w'] * frame_data['face_h']

        # Gaze displacement
        if 'gazePoint_x' in frame_data and 'gazePoint_y' in frame_data:
            center_x, center_y = 320, 240  # Adjust based on your camera resolution
            frame_data['gaze_displacement'] = np.sqrt(
                (frame_data['gazePoint_x'] - center_x) ** 2 +
                (frame_data['gazePoint_y'] - center_y) ** 2
            )

        # Head rotation magnitude
        if all(k in frame_data for k in ['head_pitch', 'head_yaw', 'head_roll']):
            frame_data['head_rotation_magnitude'] = np.sqrt(
                frame_data['head_pitch'] ** 2 +
                frame_data['head_yaw'] ** 2 +
                frame_data['head_roll'] ** 2
            )

        # Extreme angles
        if 'head_yaw' in frame_data:
            frame_data['extreme_yaw'] = int(abs(frame_data['head_yaw']) > 0.3)
        if 'head_pitch' in frame_data:
            frame_data['extreme_pitch'] = int(abs(frame_data['head_pitch']) > 0.3)

        # Multiple faces
        if 'no_of_face' in frame_data:
            frame_data['multiple_faces'] = int(frame_data['no_of_face'] > 1)

        # Hands present
        if 'hand_count' in frame_data:
            frame_data['hands_present'] = int(frame_data['hand_count'] > 0)

        return frame_data

    def _log_suspicious_events(self, frame_data: Dict):
        """Log individual suspicious events."""
        suspicious = []

        if frame_data.get('phone_present', 0) == 1:
            suspicious.append({
                'type': 'phone_detected',
                'confidence': frame_data.get('phone_conf', 0),
                'timestamp': datetime.now().isoformat(),
                'frame': self.frame_count
            })

        if frame_data.get('multiple_faces', 0) == 1:
            suspicious.append({
                'type': 'multiple_faces',
                'count': frame_data.get('no_of_face', 0),
                'timestamp': datetime.now().isoformat(),
                'frame': self.frame_count
            })

        if frame_data.get('gaze_on_script', 1) == 0:
            suspicious.append({
                'type': 'gaze_off_screen',
                'direction': frame_data.get('gaze_direction', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'frame': self.frame_count
            })

        if frame_data.get('extreme_yaw', 0) == 1 or frame_data.get('extreme_pitch', 0) == 1:
            suspicious.append({
                'type': 'extreme_head_movement',
                'yaw': frame_data.get('head_yaw', 0),
                'pitch': frame_data.get('head_pitch', 0),
                'timestamp': datetime.now().isoformat(),
                'frame': self.frame_count
            })

        self.suspicious_events_log.extend(suspicious)

        # Keep only recent events (last 1000)
        if len(self.suspicious_events_log) > 1000:
            self.suspicious_events_log = self.suspicious_events_log[-1000:]

    def _process_detection_result(self, result: Dict, frame_data: Dict) -> Dict:
        """Process and enrich detection result."""
        # Create comprehensive result
        detection_result = {
            'timestamp': result['timestamp'],
            'frame_number': self.frame_count,
            'session_id': self.current_session_id,

            # Detection results
            'is_cheating': result['is_cheating'],
            'confidence': float(result['confidence']),
            'raw_confidence': float(result['raw_confidence']),
            'alert_level': result['alert_level'],

            # Current frame info
            'current_frame_info': {
                'face_present': frame_data.get('face_present', 0),
                'no_of_faces': frame_data.get('no_of_face', 0),
                'phone_present': frame_data.get('phone_present', 0),
                'gaze_on_script': frame_data.get('gaze_on_script', 0),
                'head_yaw': frame_data.get('head_yaw', 0),
                'head_pitch': frame_data.get('head_pitch', 0),
            },

            # Contextual information
            'recent_suspicious_events': self._get_recent_suspicious_events(window=30),

            # Recommendations
            'recommended_action': self._get_recommended_action(result),
        }

        # Log alert if cheating detected
        if result['is_cheating']:
            self.alerts_history.append(detection_result)
            # Keep only recent alerts
            if len(self.alerts_history) > 100:
                self.alerts_history = self.alerts_history[-100:]

        return detection_result

    def _get_recent_suspicious_events(self, window: int = 30) -> Dict:
        """Get summary of recent suspicious events."""
        recent_events = [
            event for event in self.suspicious_events_log
            if self.frame_count - event['frame'] <= window
        ]

        summary = {
            'phone_detections': len([e for e in recent_events if e['type'] == 'phone_detected']),
            'multiple_faces': len([e for e in recent_events if e['type'] == 'multiple_faces']),
            'gaze_off_screen': len([e for e in recent_events if e['type'] == 'gaze_off_screen']),
            'extreme_movements': len([e for e in recent_events if e['type'] == 'extreme_head_movement']),
            'total_events': len(recent_events)
        }

        return summary

    def _get_recommended_action(self, result: Dict) -> str:
        """Get recommended action based on detection result."""
        if result['alert_level'] == 'HIGH':
            return "IMMEDIATE_REVIEW_REQUIRED: Strong evidence of cheating detected. Proctor should intervene immediately."
        elif result['alert_level'] == 'MEDIUM':
            return "MONITOR_CLOSELY: Suspicious behavior detected. Continue monitoring and collect evidence."
        elif result['alert_level'] == 'LOW':
            return "WATCH: Minor suspicious activity. Keep watching but no immediate action needed."
        else:
            return "NORMAL: No suspicious activity detected. Continue routine monitoring."

    def get_session_summary(self) -> Dict:
        """Get summary statistics for the current session."""
        return {
            'session_id': self.current_session_id,
            'total_frames': self.frame_count,
            'total_alerts': len(self.alerts_history),
            'alert_breakdown': {
                'high': len([a for a in self.alerts_history if a['alert_level'] == 'HIGH']),
                'medium': len([a for a in self.alerts_history if a['alert_level'] == 'MEDIUM']),
                'low': len([a for a in self.alerts_history if a['alert_level'] == 'LOW']),
            },
            'suspicious_events_summary': {
                'phone_detections': len([e for e in self.suspicious_events_log if e['type'] == 'phone_detected']),
                'multiple_faces': len([e for e in self.suspicious_events_log if e['type'] == 'multiple_faces']),
                'gaze_off_screen': len([e for e in self.suspicious_events_log if e['type'] == 'gaze_off_screen']),
                'extreme_movements': len(
                    [e for e in self.suspicious_events_log if e['type'] == 'extreme_head_movement']),
            }
        }

    def reset_session(self, new_session_id: Optional[str] = None):
        """Reset for a new exam session."""
        self.current_session_id = new_session_id
        self.frame_count = 0
        self.alerts_history = []
        self.suspicious_events_log = []
        self.detector.reset()
        print(f"Session reset. New session: {new_session_id}")


# Example usage and integration demo
def demo_real_time_monitoring():
    """
    Demonstrate real-time monitoring with simulated frames.

    In production, you would integrate this with your actual MediaPipe
    and browser telemetry data stream.
    """
    print("\n" + "=" * 80)
    print("REAL-TIME CHEATING DETECTION DEMO")
    print("=" * 80 + "\n")

    # Initialize monitor with trained model
    # Update model_path to your actual trained model
    monitor = LiveProctoringMonitor(
        model_path='models/lstm_cheating_detector_model.keras',
        model_type='lstm',
        buffer_size=30,
        confidence_threshold=0.7
    )

    print("\n>>> Simulating incoming frames...")
    print("(In production, these would come from your MediaPipe + telemetry system)\n")

    # Simulate 100 frames
    for frame_num in range(100):
        # Simulate frame data (replace with your actual MediaPipe output)
        frame_data = {
            'session_id': 'exam_session_001',
            'face_present': 1,
            'no_of_face': 1 if frame_num < 80 else 2,  # Multiple faces after frame 80
            'face_x': 260 + np.random.randn(),
            'face_y': 312 + np.random.randn(),
            'face_w': 150 + np.random.randn(),
            'face_h': 150 + np.random.randn(),
            'face_conf': 88 + np.random.randn(),
            'head_yaw': 0.015 + np.random.randn() * 0.01,
            'head_pitch': 0.018 + np.random.randn() * 0.01,
            'head_roll': -0.0005 + np.random.randn() * 0.001,
            'head_pose': 'forward',
            'gaze_on_script': 1 if frame_num < 60 else 0,  # Gaze off after frame 60
            'gaze_direction': 'center' if frame_num < 60 else 'bottom_right',
            'gazePoint_x': 336 + np.random.randn() * 10,
            'gazePoint_y': 325 + np.random.randn() * 10,
            'phone_present': 1 if frame_num > 50 and frame_num < 70 else 0,  # Phone detected 50-70
            'phone_conf': 85 if frame_num > 50 and frame_num < 70 else 0,
            'hand_count': 0,
            'hand_obj_interaction': 0,
        }

        # Process frame
        result = monitor.process_frame(frame_data)

        # Print results for key frames
        if frame_num in [30, 55, 65, 85, 99]:
            print(f"\n--- Frame {frame_num} ---")
            if result['status'] == 'buffering':
                print(f"Status: {result['message']}")
            else:
                print(f"Alert Level: {result['alert_level']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Is Cheating: {result['is_cheating']}")
                print(f"Recommended Action: {result['recommended_action']}")
                print(f"Recent Events: {result['recent_suspicious_events']}")

    # Get session summary
    print("\n" + "=" * 80)
    print("SESSION SUMMARY")
    print("=" * 80)
    summary = monitor.get_session_summary()
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print("DEMO COMPLETED")
    print("=" * 80 + "\n")


def integration_example():
    """
    Show how to integrate with your existing proctoring system.
    """
    print("\n" + "=" * 80)
    print("INTEGRATION EXAMPLE")
    print("=" * 80 + "\n")

    code_example = """
# In your main proctoring application:

from real_time_inference import LiveProctoringMonitor

# Initialize once at the start of exam
monitor = LiveProctoringMonitor(
    model_path='models/xgboost_cheating_detector.pkl',
    model_type='xgboost',
    buffer_size=30,
    confidence_threshold=0.7
)

# In your video processing loop (called for each frame)
def process_video_frame(mediapipe_landmarks, browser_telemetry):
    # Combine your MediaPipe and browser data
    frame_data = {
        'face_present': mediapipe_landmarks['face_present'],
        'no_of_face': mediapipe_landmarks['face_count'],
        'head_yaw': mediapipe_landmarks['head_rotation']['yaw'],
        'head_pitch': mediapipe_landmarks['head_rotation']['pitch'],
        'head_roll': mediapipe_landmarks['head_rotation']['roll'],
        'gaze_on_script': mediapipe_landmarks['gaze_on_screen'],
        'gaze_direction': mediapipe_landmarks['gaze_direction'],
        'gazePoint_x': mediapipe_landmarks['gaze_point'][0],
        'gazePoint_y': mediapipe_landmarks['gaze_point'][1],
        'phone_present': mediapipe_landmarks.get('phone_detected', 0),
        'phone_conf': mediapipe_landmarks.get('phone_confidence', 0),
        'hand_count': mediapipe_landmarks['hand_count'],
        # ... more features
    }

    # Get detection result
    result = monitor.process_frame(frame_data)

    # Handle result
    if result.get('is_cheating', False):
        if result['alert_level'] == 'HIGH':
            # Send immediate alert to proctor
            send_alert_to_proctor(
                student_id=current_student_id,
                alert_type='HIGH_PRIORITY',
                confidence=result['confidence'],
                evidence=result['current_frame_info']
            )
            # Maybe pause exam
            # pause_exam()

        elif result['alert_level'] == 'MEDIUM':
            # Log for review
            log_suspicious_activity(result)
            # Show warning to student
            show_warning_banner()

    # Update UI with current status
    update_monitoring_dashboard(result)

    return result

# At end of exam
def finish_exam():
    summary = monitor.get_session_summary()
    save_exam_report(summary)
    monitor.reset_session()
    """

    print(code_example)
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Run demo
    # demo_real_time_monitoring()

    # Show integration example
    integration_example()

    print("NOTE: To run the actual demo, uncomment demo_real_time_monitoring()")
    print("      and make sure you have a trained model available.")