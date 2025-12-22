"""
Batch Session Analyzer
======================

This module processes entire exam sessions at once (all frames together)
to determine if cheating occurred during the exam.

Use cases:
- Post-exam analysis (after student finishes)
- Batch processing of recorded exams
- Review flagged sessions
- Generate comprehensive exam reports
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

from app.inference.face_model_manager import FaceModelManager
from app.prediction.utils import DataPreprocessor, FeatureEngineer


class ExamSessionAnalyzer:
    """
    Analyzes complete exam sessions to determine if cheating occurred.
    Processes all frames at once and provides comprehensive verdict.
    """

    def __init__(self, model_path: str, model_type: str = 'lstm',
                 window_size: int = 30):
        """
        Initialize the session analyzer.

        Args:
            model_path: Path to trained model
            model_type: 'xgboost' or 'lstm'
            window_size: Window size used during training
        """
        self.model_type = model_type
        self.window_size = window_size

        # Load model
        print(f"Loading {model_type} model from {model_path}...")
        # if model_type == 'xgboost':
        #     self.model = XGBoostCheatingDetector(window_size=window_size)
        #     self.model.load(model_path)
        # elif model_type == 'lstm':
        self.model = FaceModelManager.get_instance().get_predictor()
        # else:
        #     raise ValueError(f"Unknown model type: {model_type}")

        # Initialize processors
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer(window_size=window_size)

        print("Session analyzer initialized successfully!")

    def analyze_session(self, frames_data: List[Dict],
                        session_id: str = None,
                        student_id: str = None,
                        aggregation_method: str = 'max',
                        confidence_threshold: float = 0.7) -> Dict:
        """
        Analyze an entire exam session to determine if cheating occurred.

        Args:
            frames_data: List of dictionaries, each containing frame features
                Example: [
                    {'face_present': 1, 'phone_present': 0, ...},
                    {'face_present': 1, 'phone_present': 1, ...},
                    ...
                ]
            session_id: Unique identifier for this exam session
            student_id: Student identifier
            aggregation_method: How to aggregate predictions across windows
                - 'max': Conservative - any window shows cheating = cheating
                - 'mean': Average confidence across all windows
                - 'percentage': Percentage of windows flagged as cheating
                - 'weighted': Weight recent windows more heavily
            confidence_threshold: Threshold for final verdict

        Returns:
            Dictionary with comprehensive analysis results
        """
        if not frames_data:
            raise ValueError("frames_data cannot be empty")

        print(f"\nAnalyzing exam session with {len(frames_data)} frames...")

        # Convert to DataFrame
        df = pd.DataFrame(frames_data)

        # Add session metadata
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if student_id is None:
            student_id = "unknown_student"

        df['session_id'] = session_id
        # df['student_id'] = student_id
        df['row_id'] = range(len(df))

        # Preprocess
        df = self.preprocessor.clean_data(df)

        # Feature engineering
        df = self.feature_engineer.create_temporal_features(df)
        df = self.feature_engineer.create_derived_features(df)

        # Get predictions based on model type
        if self.model_type == 'xgboost':
            window_predictions, window_confidences = self._analyze_with_xgboost(df)
        else:
            window_predictions, window_confidences = self._analyze_with_lstm(df)

        # Aggregate results
        final_verdict = self._aggregate_predictions(
            window_predictions,
            window_confidences,
            method=aggregation_method,
            threshold=confidence_threshold
        )

        # Analyze suspicious events
        suspicious_events = self._extract_suspicious_events(df)

        # Create timeline
        timeline = self._create_timeline(
            df, window_predictions, window_confidences
        )

        # Generate comprehensive report
        report = {
            'session_info': {
                'session_id': session_id,
                'student_id': student_id,
                'total_frames': len(df),
                'duration_seconds': len(df),  # Assuming 1 fps
                'analysis_timestamp': datetime.now().isoformat()
            },

            'verdict': {
                'is_cheating': final_verdict['is_cheating'],
                'confidence': final_verdict['confidence'],
                'confidence_level': final_verdict['confidence_level'],
                'verdict_basis': final_verdict['basis']
            },

            'window_analysis': {
                'total_windows': len(window_predictions),
                'windows_flagged': int(sum(window_predictions)),
                'percentage_flagged': (
                            sum(window_predictions) / len(window_predictions) * 100) if window_predictions else 0,
                'max_confidence': float(max(window_confidences)) if window_confidences else 0,
                'mean_confidence': float(np.mean(window_confidences)) if window_confidences else 0,
                'predictions_per_window': [
                    {
                        'window_id': i,
                        'is_cheating': bool(pred),
                        'confidence': float(conf),
                        'start_frame': i * self.window_size,
                        'end_frame': min((i + 1) * self.window_size, len(df))
                    }
                    for i, (pred, conf) in enumerate(zip(window_predictions, window_confidences))
                ]
            },

            'suspicious_events': suspicious_events,

            'timeline': timeline,

            'recommendations': self._get_recommendations(final_verdict, suspicious_events)
        }

        return report

    def _analyze_with_xgboost(self, df: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """Analyze using XGBoost model."""
        # Create window features
        window_df = self.feature_engineer.create_window_features(
            df, window_size=self.window_size
        )

        if len(window_df) == 0:
            print("Warning: Not enough frames to create windows")
            return [], []

        # Prepare features
        X, _ = self.model.prepare_features(window_df)

        # Get predictions
        predictions = self.model.predict(X, threshold=0.5)
        confidences = self.model.predict_proba(X)

        return predictions.tolist(), confidences.tolist()

    def _analyze_with_lstm(self, df: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """Analyze using LSTM model."""
        # Create sequences
        X, _ = self.model.create_sequences(df, self.model.feature_columns)

        if len(X) == 0:
            print("Warning: Not enough frames to create sequences")
            return [], []

        # Get predictions
        predictions = self.model.predict(X, threshold=0.5)
        confidences = self.model.predict_proba(X)

        return predictions.tolist(), confidences.tolist()

    def _aggregate_predictions(self, predictions: List[int],
                               confidences: List[float],
                               method: str,
                               threshold: float) -> Dict:
        """
        Aggregate window/sequence predictions into final verdict.
        """
        if not predictions:
            return {
                'is_cheating': False,
                'confidence': 0.0,
                'confidence_level': 'NONE',
                'basis': 'Insufficient data for analysis'
            }

        if method == 'max':
            # Conservative: if ANY window is flagged, consider cheating
            final_confidence = max(confidences)
            is_cheating = final_confidence >= threshold
            basis = f"Highest confidence window: {final_confidence:.2f}"

        elif method == 'mean':
            # Average confidence across all windows
            final_confidence = np.mean(confidences)
            is_cheating = final_confidence >= threshold
            basis = f"Average confidence across {len(confidences)} windows: {final_confidence:.2f}"

        elif method == 'percentage':
            # Percentage of windows flagged
            percentage_flagged = sum(predictions) / len(predictions)
            final_confidence = percentage_flagged
            is_cheating = percentage_flagged >= threshold
            basis = f"{percentage_flagged * 100:.1f}% of windows flagged as cheating"

        elif method == 'weighted':
            # Weight recent windows more heavily (decay factor)
            weights = np.exp(np.linspace(-2, 0, len(confidences)))
            weights = weights / weights.sum()
            final_confidence = np.sum(np.array(confidences) * weights)
            is_cheating = final_confidence >= threshold
            basis = f"Weighted confidence (recent emphasis): {final_confidence:.2f}"

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Determine confidence level
        if final_confidence >= 0.9:
            confidence_level = "VERY_HIGH"
        elif final_confidence >= 0.75:
            confidence_level = "HIGH"
        elif final_confidence >= 0.6:
            confidence_level = "MEDIUM"
        elif final_confidence >= 0.4:
            confidence_level = "LOW"
        else:
            confidence_level = "VERY_LOW"

        return {
            'is_cheating': is_cheating,
            'confidence': float(final_confidence),
            'confidence_level': confidence_level,
            'basis': basis
        }

    def _extract_suspicious_events(self, df: pd.DataFrame) -> Dict:
        """Extract and summarize suspicious events from the session."""
        events = {
            'phone_detections': {
                'count': int((df.get('phone_present', 0) == 1).sum()),
                'frames': df[df.get('phone_present', 0) == 1].index.tolist()[:10],  # First 10
                'max_confidence': float(df.get('phone_conf', 0).max()) if 'phone_conf' in df else 0
            },
            'multiple_faces': {
                'count': int((df.get('no_of_face', 1) > 1).sum()),
                'frames': df[df.get('no_of_face', 1) > 1].index.tolist()[:10],
                'max_faces': int(df.get('no_of_face', 1).max()) if 'no_of_face' in df else 1
            },
            'gaze_off_screen': {
                'count': int((df.get('gaze_on_script', 1) == 0).sum()),
                'frames': df[df.get('gaze_on_script', 1) == 0].index.tolist()[:10],
                'percentage': float((df.get('gaze_on_script', 1) == 0).mean() * 100) if 'gaze_on_script' in df else 0
            },
            'extreme_head_movements': {
                'extreme_yaw_count': int((np.abs(df.get('head_yaw', 0)) > 0.3).sum()) if 'head_yaw' in df else 0,
                'extreme_pitch_count': int((np.abs(df.get('head_pitch', 0)) > 0.3).sum()) if 'head_pitch' in df else 0,
                'frames': df[(np.abs(df.get('head_yaw', 0)) > 0.3) |
                             (np.abs(df.get('head_pitch', 0)) > 0.3)].index.tolist()[:10]
            },
            'no_face_detected': {
                'count': int((df.get('face_present', 1) == 0).sum()),
                'frames': df[df.get('face_present', 1) == 0].index.tolist()[:10]
            }
        }

        # Add summary
        total_suspicious = sum([
            events['phone_detections']['count'],
            events['multiple_faces']['count'],
            events['extreme_head_movements']['extreme_yaw_count'] +
            events['extreme_head_movements']['extreme_pitch_count']
        ])

        events['summary'] = {
            'total_suspicious_events': total_suspicious,
            'frames_with_issues': len(set(
                events['phone_detections']['frames'] +
                events['multiple_faces']['frames'] +
                events['extreme_head_movements']['frames']
            ))
        }

        return events

    def _create_timeline(self, df: pd.DataFrame,
                         predictions: List[int],
                         confidences: List[float]) -> List[Dict]:
        """Create a timeline of significant events during the exam."""
        timeline = []

        # Add high-confidence cheating windows
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            if conf >= 0.7:
                start_frame = i * self.window_size
                end_frame = min((i + 1) * self.window_size, len(df))

                # Extract what was suspicious in this window
                window_df = df.iloc[start_frame:end_frame]

                suspicious_in_window = []
                if (window_df.get('phone_present', 0) == 1).any():
                    suspicious_in_window.append('phone_detected')
                if (window_df.get('no_of_face', 1) > 1).any():
                    suspicious_in_window.append('multiple_faces')
                if (window_df.get('gaze_on_script', 1) == 0).mean() > 0.5:
                    suspicious_in_window.append('gaze_off_screen')

                timeline.append({
                    'window_id': i,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time_seconds': start_frame,
                    'end_time_seconds': end_frame,
                    'confidence': float(conf),
                    'predicted_cheating': bool(pred),
                    'suspicious_behaviors': suspicious_in_window
                })

        return timeline

    def _get_recommendations(self, verdict: Dict, suspicious_events: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if verdict['is_cheating']:
            if verdict['confidence_level'] in ['VERY_HIGH', 'HIGH']:
                recommendations.append(
                    "IMMEDIATE REVIEW REQUIRED: Strong evidence of cheating detected. "
                    "Manual review by proctor is essential before finalizing any decisions."
                )
            else:
                recommendations.append(
                    "REVIEW RECOMMENDED: Suspicious behavior detected. "
                    "Consider manual review of flagged timestamps."
                )

        # Specific recommendations based on events
        if suspicious_events['phone_detections']['count'] > 5:
            recommendations.append(
                f"Phone detected {suspicious_events['phone_detections']['count']} times. "
                f"Review frames: {suspicious_events['phone_detections']['frames'][:5]}"
            )

        if suspicious_events['multiple_faces']['count'] > 3:
            recommendations.append(
                f"Multiple people detected in {suspicious_events['multiple_faces']['count']} frames. "
                f"Review frames: {suspicious_events['multiple_faces']['frames'][:5]}"
            )

        if suspicious_events['gaze_off_screen']['percentage'] > 50:
            recommendations.append(
                f"Student's gaze was off-screen {suspicious_events['gaze_off_screen']['percentage']:.1f}% of the time. "
                "This may indicate looking at external resources."
            )

        if not recommendations:
            recommendations.append(
                "No significant suspicious behavior detected. Exam appears normal."
            )

        return recommendations

    def analyze_batch_sessions(self, sessions_data: List[Dict],
                               confidence_threshold: float = 0.7) -> pd.DataFrame:
        """
        Analyze multiple exam sessions at once.

        Args:
            sessions_data: List of session dictionaries, each containing:
                {
                    'session_id': 'session_123',
                    'student_id': 'student_456',
                    'frames': [frame1, frame2, ...]
                }
            confidence_threshold: Threshold for final verdict

        Returns:
            DataFrame with results for all sessions
        """
        print(f"\nAnalyzing {len(sessions_data)} exam sessions in batch...")

        results = []

        for i, session in enumerate(sessions_data, 1):
            print(f"\nProcessing session {i}/{len(sessions_data)}: {session.get('session_id', 'unknown')}")

            try:
                report = self.analyze_session(
                    frames_data=session['frames'],
                    session_id=session.get('session_id'),
                    student_id=session.get('student_id'),
                    confidence_threshold=confidence_threshold
                )

                results.append({
                    'session_id': report['session_info']['session_id'],
                    'student_id': report['session_info']['student_id'],
                    'total_frames': report['session_info']['total_frames'],
                    'is_cheating': report['verdict']['is_cheating'],
                    'confidence': report['verdict']['confidence'],
                    'confidence_level': report['verdict']['confidence_level'],
                    'windows_flagged': report['window_analysis']['windows_flagged'],
                    'percentage_flagged': report['window_analysis']['percentage_flagged'],
                    'phone_detections': report['suspicious_events']['phone_detections']['count'],
                    'multiple_faces': report['suspicious_events']['multiple_faces']['count'],
                    'gaze_off_pct': report['suspicious_events']['gaze_off_screen']['percentage'],
                    'recommendations': '; '.join(report['recommendations'])
                })

            except Exception as e:
                print(f"Error processing session {session.get('session_id')}: {e}")
                results.append({
                    'session_id': session.get('session_id', 'unknown'),
                    'student_id': session.get('student_id', 'unknown'),
                    'error': str(e)
                })

        return pd.DataFrame(results)

    def save_report(self, report: Dict, filepath: str):
        """Save analysis report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {filepath}")

    def export_summary_csv(self, report: Dict, filepath: str):
        """Export session summary to CSV."""
        summary_data = {
            'session_id': report['session_info']['session_id'],
            'student_id': report['session_info']['student_id'],
            'total_frames': report['session_info']['total_frames'],
            'is_cheating': report['verdict']['is_cheating'],
            'confidence': report['verdict']['confidence'],
            'confidence_level': report['verdict']['confidence_level'],
            'windows_flagged': report['window_analysis']['windows_flagged'],
            'phone_detections': report['suspicious_events']['phone_detections']['count'],
            'multiple_faces': report['suspicious_events']['multiple_faces']['count'],
            'analysis_timestamp': report['session_info']['analysis_timestamp']
        }

        df = pd.DataFrame([summary_data])
        df.to_csv(filepath, index=False)
        print(f"Summary exported to: {filepath}")




if __name__ == "__main__":
    print("Batch Session Analyzer")
    print("=" * 80)
    print("\nThis module allows you to analyze complete exam sessions")
    print("by passing all frames at once (as an array/list).\n")

    # Uncomment to run examples:
    # example_batch_analysis()
    # example_multiple_sessions()