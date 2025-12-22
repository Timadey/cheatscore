"""
Exam Proctoring Cheating Detection System
==========================================

This module provides a complete implementation for detecting cheating behavior
during online exams using temporal features and machine learning models.

Supports:
- Data preprocessing and feature engineering
- Sliding window approach with XGBoost
- LSTM-based sequential model
- Real-time inference
- Model interpretability with SHAP
"""

import numpy as np
import pandas as pd
# from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# ML Libraries
# from sklearn.model_selection import train_test_split, TimeSeriesSplit
# from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, fbeta_score
)
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE

# import xgboost as xgb
# from xgboost import XGBClassifier

# Deep Learning
# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks
# from tensorflow.keras.utils import to_categorical

# Visualization and Interpretability
# import matplotlib.pyplot as plt
# import seaborn as sns

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

# Utilities
import joblib


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for proctoring data.
    """

    def __init__(self):
        self.scaler = None
        self.feature_columns = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load proctoring data from CSV file."""
        df = pd.read_csv(filepath, sep='\t')
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess raw data."""
        df = df.copy()

        # Handle missing values
        # For numerical features, fill with 0 (represents absence)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(0)

        # For categorical features like gaze_direction, fill with 'unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'label']
        df[categorical_cols] = df[categorical_cols].fillna('unknown')

        # Handle invalid values (negative coordinates should be 0)
        coord_cols = [col for col in df.columns if any(x in col for x in ['_x', '_y', '_w', '_h'])]
        for col in coord_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # Convert head_pose to numerical if it's categorical
        if df['head_pose'].dtype == 'object':
            pose_mapping = {
                'forward': 0, 'left': 1, 'right': 2,
                'up': 3, 'down': 4, 'None': -1, 'unknown': -1
            }
            df['head_pose_encoded'] = df['head_pose'].map(pose_mapping).fillna(-1)

        # Convert gaze_direction to numerical
        if df['gaze_direction'].dtype == 'object':
            gaze_mapping = {
                'center': 0, 'left': 1, 'right': 2,
                'top_left': 3, 'top_right': 4,
                'bottom_left': 5, 'bottom_right': 6,
                'None': -1, 'unknown': -1
            }
            df['gaze_direction_encoded'] = df['gaze_direction'].map(gaze_mapping).fillna(-1)

        return df

    def add_student_session_id(self, df: pd.DataFrame,
                               session_break_threshold: int = 300) -> pd.DataFrame:
        """
        Add student and session identifiers if not present.
        If your data has these, this can be skipped.
        """
        df = df.copy()

        # FIX: Ensure multiple unique session IDs are generated if 'session_id' is not present,
        # to allow train_test_split to work correctly by sessions.
        if 'session_id' not in df.columns:
            num_frames = len(df)
            min_sessions_for_split = 5  # Aim for at least this many sessions for train_test_split
            # Distribute frames into a few arbitrary sessions
            frames_per_session_block = max(1, num_frames // min_sessions_for_split)
            df['session_id'] = (df.index // frames_per_session_block).astype(int)
            print(
                f"INFO: 'session_id' column not found. Created {df['session_id'].nunique()} dummy session IDs for splitting.")

        if 'row_id' not in df.columns:  # ensure row_id is present for temporal features if not already
            df['row_id'] = range(len(df))

        if 'student_id' not in df.columns:
            df['student_id'] = 'student_1'  # Single student for now, adjust if multiple students exist

        return df


class FeatureEngineer:
    """
    Creates temporal and derived features for cheating detection.
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of frames to aggregate (e.g., 30 frames = 30 seconds at 1fps)
        """
        self.window_size = window_size

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on temporal patterns."""
        df = df.copy()

        # Sort by session and time
        if 'row_id' in df.columns:
            df = df.sort_values(['session_id', 'row_id'])

        # Rolling window features (within each session)
        rolling_cols = {
            'gaze_on_script': ['mean', 'std', 'min'],
            'phone_present': ['sum', 'mean'],
            'no_of_face': ['mean', 'max'],
            'head_yaw': ['mean', 'std', 'max', 'min'],
            'head_pitch': ['mean', 'std', 'max', 'min'],
            'hand_count': ['mean', 'max'],
        }

        for col, aggs in rolling_cols.items():
            if col in df.columns:
                for agg in aggs:
                    rolling = df.groupby('session_id')[col].rolling(
                        window=self.window_size, min_periods=1
                    ).agg(agg)
                    df[f'{col}_rolling_{agg}'] = rolling.values

        # Change detection (frame-to-frame changes)
        change_cols = ['head_yaw', 'head_pitch', 'head_roll', 'gazePoint_x', 'gazePoint_y']
        for col in change_cols:
            if col in df.columns:
                df[f'{col}_change'] = df.groupby('session_id')[col].diff().abs()
                df[f'{col}_change'] = df[f'{col}_change'].fillna(0)

        # Cumulative suspicious events
        if 'phone_present' in df.columns:
            df['phone_cumsum'] = df.groupby('session_id')['phone_present'].cumsum()

        # Time since last suspicious event
        if 'phone_present' in df.columns:
            df['frames_since_phone'] = 0
            for session in df['session_id'].unique():
                mask = df['session_id'] == session
                phone_events = df.loc[mask, 'phone_present'].values
                frames_since = np.zeros(len(phone_events))
                counter = 0
                for i, val in enumerate(phone_events):
                    if val == 1:
                        counter = 0
                    else:
                        counter += 1
                    frames_since[i] = counter
                df.loc[mask, 'frames_since_phone'] = frames_since

        return df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from raw measurements."""
        df = df.copy()

        # Face size (potential indicator of distance from camera)
        if all(col in df.columns for col in ['face_w', 'face_h']):
            df['face_area'] = df['face_w'] * df['face_h']

        # Eye distance (sanity check for face detection quality)
        if all(col in df.columns for col in ['left_eye_x', 'right_eye_x', 'left_eye_y', 'right_eye_y']):
            df['eye_distance'] = np.sqrt(
                (df['left_eye_x'] - df['right_eye_x']) ** 2 +
                (df['left_eye_y'] - df['right_eye_y']) ** 2
            )

        # Gaze displacement from center (assuming center is around image center)
        if all(col in df.columns for col in ['gazePoint_x', 'gazePoint_y']):
            # Assuming 640x480 image, adjust as needed
            center_x, center_y = 320, 240
            df['gaze_displacement'] = np.sqrt(
                (df['gazePoint_x'] - center_x) ** 2 +
                (df['gazePoint_y'] - center_y) ** 2
            )

        # Head pose magnitude (total head rotation)
        if all(col in df.columns for col in ['head_pitch', 'head_yaw', 'head_roll']):
            df['head_rotation_magnitude'] = np.sqrt(
                df['head_pitch'] ** 2 + df['head_yaw'] ** 2 + df['head_roll'] ** 2
            )

        # Extreme head angles
        if 'head_yaw' in df.columns:
            df['extreme_yaw'] = (np.abs(df['head_yaw']) > 0.3).astype(int)
        if 'head_pitch' in df.columns:
            df['extreme_pitch'] = (np.abs(df['head_pitch']) > 0.3).astype(int)

        # Multiple faces detected (strong cheating signal)
        if 'no_of_face' in df.columns:
            df['multiple_faces'] = (df['no_of_face'] > 1).astype(int)

        # Hand near face (potential earpiece or hidden device)
        if 'hand_count' in df.columns:
            df['hands_present'] = (df['hand_count'] > 0).astype(int)

        return df

    def create_window_features(self, df: pd.DataFrame,
                               window_size: int = None, label_threshold: float = 0.7) -> pd.DataFrame:
        """
        Create aggregated features over sliding windows.
        Returns one row per window instead of per frame.
        """
        if window_size is None:
            window_size = self.window_size

        df = df.copy()

        # Group by session
        window_features = []

        for session_id in df['session_id'].unique():
            session_df = df[df['session_id'] == session_id].copy()

            # Create windows
            num_windows = len(session_df) // window_size

            for i in range(num_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                window_df = session_df.iloc[start_idx:end_idx]

                # Calculate suspicious frame count
                suspicious_frame_count = window_df['label'].sum()
                suspicious_frame_pct = window_df['label'].mean()

                # Calculate strong evidence signals
                phone_frames = window_df['phone_present'].sum()
                multiple_face_frames = (window_df['no_of_face'] > 1).sum()
                gaze_off_frames = (1 - window_df['gaze_on_script']).sum()

                # FIXED LABELING LOGIC
                # Require SUSTAINED suspicious behavior + strong evidence
                is_cheating = (
                    # Option 1: Threshold-based
                        suspicious_frame_pct >= label_threshold

                        # OR Option 2: Evidence-based
                        or (suspicious_frame_count >= 20 and (
                        phone_frames >= 15 or  # Phone for 15+ seconds
                        multiple_face_frames >= 15 or  # Multiple faces 15+ seconds
                        gaze_off_frames >= 25  # Gaze off for 25+ seconds
                ))
                )

                # Aggregate features
                features = {
                    'session_id': session_id,
                    'window_id': i,

                    # Gaze features
                    'gaze_off_script_pct': 1 - window_df[
                        'gaze_on_script'].mean() if 'gaze_on_script' in window_df else 0,
                    'gaze_off_script_duration': (
                                1 - window_df['gaze_on_script']).sum() if 'gaze_on_script' in window_df else 0,

                    # Phone detection
                    'phone_detected_count': window_df['phone_present'].sum() if 'phone_present' in window_df else 0,
                    'phone_detected_pct': window_df['phone_present'].mean() if 'phone_present' in window_df else 0,
                    'avg_phone_conf': window_df[window_df['phone_present'] == 1][
                        'phone_conf'].mean() if 'phone_conf' in window_df else 0,

                    # Face detection
                    'multiple_faces_count': (window_df['no_of_face'] > 1).sum() if 'no_of_face' in window_df else 0,
                    'no_face_count': (window_df['face_present'] == 0).sum() if 'face_present' in window_df else 0,
                    'avg_face_conf': window_df['face_conf'].mean() if 'face_conf' in window_df else 0,

                    # Head pose
                    'extreme_yaw_count': (np.abs(window_df['head_yaw']) > 0.3).sum() if 'head_yaw' in window_df else 0,
                    'extreme_pitch_count': (
                                np.abs(window_df['head_pitch']) > 0.3).sum() if 'head_pitch' in window_df else 0,
                    'avg_head_yaw': window_df['head_yaw'].mean() if 'head_yaw' in window_df else 0,
                    'std_head_yaw': window_df['head_yaw'].std() if 'head_yaw' in window_df else 0,
                    'avg_head_pitch': window_df['head_pitch'].mean() if 'head_pitch' in window_df else 0,
                    'std_head_pitch': window_df['head_pitch'].std() if 'head_pitch' in window_df else 0,
                    'max_abs_yaw': np.abs(window_df['head_yaw']).max() if 'head_yaw' in window_df else 0,
                    'max_abs_pitch': np.abs(window_df['head_pitch']).max() if 'head_pitch' in window_df else 0,

                    # Hand detection
                    'hand_detected_count': (window_df['hand_count'] > 0).sum() if 'hand_count' in window_df else 0,
                    'avg_hand_count': window_df['hand_count'].mean() if 'hand_count' in window_df else 0,

                    # Gaze movement
                    'gaze_movement_x': window_df['gazePoint_x'].std() if 'gazePoint_x' in window_df else 0,
                    'gaze_movement_y': window_df['gazePoint_y'].std() if 'gazePoint_y' in window_df else 0,

                    # Label (majority vote or any positive)
                    # 'label': window_df['label'].max() if 'label' in window_df else 0,  # Conservative: any cheating frame = cheating window
                    'label': 1 if (window_df['label'].sum() >= 20 and  # At least 20/30 frames
                                   # TODO: add more strong signals
                                   (window_df['phone_present'].sum() > 10 or  # Strong signal
                                    window_df[window_df['no_of_face'] > 1].shape[0] > 10 or
                                    (1 - window_df['gaze_on_script']).sum() > 25)) else 0
                }

                # Fill NaN with 0
                features = {k: (0 if pd.isna(v) else v) for k, v in features.items()}
                window_features.append(features)

        return pd.DataFrame(window_features)


#
#
# def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
#                    y_proba: np.ndarray = None, model_name: str = "Model"):
#     """
#     Comprehensive model evaluation.
#     """
#     print(f"\n{'=' * 60}")
#     print(f"{model_name} Evaluation Results")
#     print(f"{'=' * 60}\n")
#
#     # Classification report
#     print("Classification Report:")
#     # FIX: Explicitly specify labels to handle cases where one class might be missing in y_true
#     print(classification_report(y_true, y_pred,
#                                 target_names=['Not Cheating', 'Cheating'],
#                                 labels=[0, 1],
#                                 digits=4))
#
#     # Confusion Matrix
#     # FIX: Explicitly specify labels to handle cases where one class might be missing in y_true
#     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
#     print("\nConfusion Matrix:")
#     print(f"{'':15} {'Predicted No':15} {'Predicted Yes':15}")
#     print(f"{'Actual No':15} {cm[0, 0]:<15} {cm[0, 1]:<15}")
#     print(f"{'Actual Yes':15} {cm[1, 0]:<15} {cm[1, 1]:<15}")
#
#     # Additional metrics
#     if y_proba is not None:
#         auc_score = roc_auc_score(y_true, y_proba)
#         print(f"\nAUC-ROC Score: {auc_score:.4f}")
#
#     # F-beta scores (F2 emphasizes recall)
#     # FIX: Specify pos_label for binary classification metrics to ensure consistent behavior
#     f1 = f1_score(y_true, y_pred, pos_label=1)
#     f2 = fbeta_score(y_true, y_pred, beta=2, pos_label=1)
#     print(f"\nF1 Score: {f1:.4f}")
#     print(f"F2 Score (emphasizes recall): {f2:.4f}")
#
#     return {
#         'classification_report': classification_report(y_true, y_pred, output_dict=True, labels=[0, 1]),
#         'confusion_matrix': cm,
#         'f1_score': f1,
#         'f2_score': f2,
#         'auc_score': auc_score if y_proba is not None else None
#     }
#
#
# def plot_training_history(history, save_path: str = None):
#     """Plot training history for LSTM model."""
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#
#     # Loss
#     axes[0, 0].plot(history.history['loss'], label='Training Loss')
#     if 'val_loss' in history.history:
#         axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
#     axes[0, 0].set_title('Model Loss')
#     axes[0, 0].set_xlabel('Epoch')
#     axes[0, 0].set_ylabel('Loss')
#     axes[0, 0].legend()
#     axes[0, 0].grid(True)
#
#     # Accuracy
#     axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
#     if 'val_accuracy' in history.history:
#         axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
#     axes[0, 1].set_title('Model Accuracy')
#     axes[0, 1].set_xlabel('Epoch')
#     axes[0, 1].set_ylabel('Accuracy')
#     axes[0, 1].legend()
#     axes[0, 1].grid(True)
#
#     # Precision
#     if 'precision' in history.history:
#         axes[1, 0].plot(history.history['precision'], label='Training Precision')
#         if 'val_precision' in history.history:
#             axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
#         axes[1, 0].set_title('Model Precision')
#         axes[1, 0].set_xlabel('Epoch')
#         axes[1, 0].set_ylabel('Precision')
#         axes[1, 0].legend()
#         axes[1, 0].grid(True)
#
#     # Recall
#     if 'recall' in history.history:
#         axes[1, 1].plot(history.history['recall'], label='Training Recall')
#         if 'val_recall' in history.history:
#             axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
#         axes[1, 1].set_title('Model Recall')
#         axes[1, 1].set_xlabel('Epoch')
#         axes[1, 1].set_ylabel('Recall')
#         axes[1, 1].legend()
#         axes[1, 1].grid(True)
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Training history plot saved to {save_path}")
#
#     plt.show()
#
#
# def plot_feature_importance(feature_importance_df: pd.DataFrame,
#                             top_n: int = 20, save_path: str = None):
#     """Plot feature importance."""
#     plt.figure(figsize=(12, 8))
#
#     top_features = feature_importance_df.head(top_n)
#     plt.barh(range(len(top_features)), top_features['importance'])
#     plt.yticks(range(len(top_features)), top_features['feature'])
#     plt.xlabel('Importance')
#     plt.title(f'Top {top_n} Feature Importances')
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Feature importance plot saved to {save_path}")
#
#     plt.show()


# Example usage and main training pipeline
if __name__ == "__main__":
    print("Exam Proctoring Cheating Detection System")
    print("=" * 60)

    # This is a template - you'll need to provide your actual data file
    print("\nNote: This is a template. Please update with your actual data file path.")
    print("Example usage is provided below.\n")