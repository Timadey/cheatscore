from datetime import datetime

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')
from app.prediction.utils import FeatureEngineer


# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, fbeta_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import xgboost as xgb
from xgboost import XGBClassifier

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

# Visualization and Interpretability
import seaborn as sns
import joblib

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")



class XGBoostCheatingDetector:
    """
    XGBoost-based cheating detection model with sliding window features.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.class_weights = None

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
        # Exclude non-feature columns
        exclude_cols = ['session_id', 'window_id', 'label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].values
        y = df['label'].values if 'label' in df.columns else None

        self.feature_columns = feature_cols

        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              use_smote: bool = True):
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            use_smote: Whether to use SMOTE for handling imbalance
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Handle class imbalance
        if use_smote and len(np.unique(y_train)) > 1:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        # Compute class weights
        classes = np.unique(y_train)
        self.class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = {classes[i]: self.class_weights[i] for i in range(len(classes))}

        sample_weights = np.array([weight_dict[label] for label in y_train])

        # Train XGBoost
        print("\nTraining XGBoost model...")

        eval_set = []
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=20 if eval_set else None
        )

        self.model.fit(
            X_train_scaled,
            y_train,
            sample_weight=sample_weights,
            eval_set=eval_set if eval_set else None,
            verbose=True
        )

        print("Training completed!")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict cheating labels."""
        X_scaled = self.scaler.transform(X)
        probas = self.model.predict_proba(X_scaled)[:, 1]
        return (probas >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get cheating probability."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return feature_importance_df.head(top_n)

    def explain_predictions(self, X: np.ndarray, num_samples: int = 100):
        """
        Generate SHAP explanations for model predictions.
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return None

        X_scaled = self.scaler.transform(X)

        # Sample data for efficiency
        if len(X_scaled) > num_samples:
            indices = np.random.choice(len(X_scaled), num_samples, replace=False)
            X_sample = X_scaled[indices]
        else:
            X_sample = X_scaled

        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        return explainer, shap_values, X_sample

    def save(self, filepath: str):
        """Save model and scaler."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'window_size': self.window_size,
            'class_weights': self.class_weights
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model and scaler."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.window_size = model_data['window_size']
        self.class_weights = model_data.get('class_weights')
        print(f"Model loaded from {filepath}")


class LSTMCheatingDetector:
    """
    LSTM-based sequential cheating detection model.
    """

    def __init__(self, sequence_length: int = 60, features_per_frame: int = 30):
        """
        Args:
            sequence_length: Number of frames in each sequence
            features_per_frame: Number of features per frame
        """
        self.sequence_length = sequence_length
        self.features_per_frame = features_per_frame
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

    def create_sequences(self, df: pd.DataFrame,
                         feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.

        Returns:
            X: Shape (num_sequences, sequence_length, num_features)
            y: Shape (num_sequences,)
        """
        sequences = []
        labels = []

        for session_id in df['session_id'].unique():
            session_df = df[df['session_id'] == session_id]

            # Create overlapping sequences
            for i in range(len(session_df) - self.sequence_length + 1):
                sequence = session_df.iloc[i:i + self.sequence_length]

                # Extract features
                X_seq = sequence[feature_cols].values

                # Label: majority vote or conservative (any positive = positive)
                # y_label = sequence['label'].max() if 'label' in sequence.columns else 0
                y_label = 1 if (sequence['label'].sum() >= 40 and  # At least 40/60 frames
                                (sequence['phone_present'].sum() > 20 or
                                 sequence[sequence['no_of_face'] > 1].shape[0] > 20)) else 0

                sequences.append(X_seq)
                labels.append(y_label)

        X = np.array(sequences)
        y = np.array(labels)

        self.feature_columns = feature_cols

        return X, y

    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture."""
        model = models.Sequential([
            # First LSTM layer
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.BatchNormalization(),

            # Second LSTM layer
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            layers.BatchNormalization(),

            # Dense layers
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(8, activation='relu'),

            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )

        self.model = model
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32):
        """
        Train LSTM model.
        """
        # Scale features (per feature, across all sequences)
        num_sequences, seq_len, num_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, num_features)
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(num_sequences, seq_len, num_features)

        if X_val is not None:
            num_val_sequences = X_val.shape[0]
            X_val_reshaped = X_val.reshape(-1, num_features)
            X_val_scaled = self.scaler.transform(X_val_reshaped)
            X_val_scaled = X_val_scaled.reshape(num_val_sequences, seq_len, num_features)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None

        # Build model if not already built
        if self.model is None:
            self.build_model(input_shape=(seq_len, num_features))

        # Class weights
        class_weight = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weight[i] for i in range(len(class_weight))}

        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        # Train
        print("\nTraining LSTM model...")
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        print("Training completed!")
        return history

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict cheating labels."""
        # Scale
        num_sequences, seq_len, num_features = X.shape
        X_reshaped = X.reshape(-1, num_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(num_sequences, seq_len, num_features)

        probas = self.model.predict(X_scaled, verbose=0).flatten()
        return (probas >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get cheating probability."""
        # Scale
        num_sequences, seq_len, num_features = X.shape
        X_reshaped = X.reshape(-1, num_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(num_sequences, seq_len, num_features)

        return self.model.predict(X_scaled, verbose=0).flatten()

    def save(self, filepath: str):
        """Save model and scaler."""
        # Save Keras model
        self.model.save(f"{filepath}_model.keras")

        # Save metadata
        metadata = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'features_per_frame': self.features_per_frame
        }
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model and metadata."""
        self.model = models.load_model(f"{filepath}_model.keras")
        metadata = joblib.load(f"{filepath}_metadata.pkl")

        self.scaler = metadata['scaler']
        self.feature_columns = metadata['feature_columns']
        self.sequence_length = metadata['sequence_length']
        self.features_per_frame = metadata['features_per_frame']
        print(f"Model loaded from {filepath}")


class RealTimeDetector:
    """
    Real-time cheating detection with buffering and smoothing.
    """

    def __init__(self, model, buffer_size: int = 30,
                 smoothing_window: int = 5,
                 confidence_threshold: float = 0.7):
        """
        Args:
            model: Trained model (XGBoost or LSTM)
            buffer_size: Number of frames to buffer
            smoothing_window: Number of predictions to smooth over
            confidence_threshold: Threshold for alerting
        """
        self.model = model
        self.buffer_size = buffer_size
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold

        self.frame_buffer = []
        self.prediction_buffer = []

    def add_frame(self, frame_data: Dict) -> Optional[Dict]:
        """
        Add new frame and get detection result.

        Args:
            frame_data: Dictionary with frame features

        Returns:
            Detection result if buffer is full, None otherwise
        """
        self.frame_buffer.append(frame_data)

        # Keep buffer size fixed
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

        # Only predict when buffer is full
        if len(self.frame_buffer) < self.buffer_size:
            return None

        # Prepare features for prediction
        if isinstance(self.model, XGBoostCheatingDetector):
            # For XGBoost: use window features
            window_df = pd.DataFrame(self.frame_buffer)
            feature_engineer = FeatureEngineer(window_size=self.buffer_size)
            window_features = feature_engineer.create_window_features(
                window_df, window_size=self.buffer_size
            )

            if len(window_features) > 0:
                X, _ = self.model.prepare_features(window_features)
                proba = self.model.predict_proba(X)[0]
            else:
                proba = 0.0

        elif isinstance(self.model, LSTMCheatingDetector):
            # For LSTM: use sequence
            window_df = pd.DataFrame(self.frame_buffer)
            X_seq = window_df[self.model.feature_columns].values
            X_seq = X_seq.reshape(1, self.buffer_size, -1)
            proba = self.model.predict_proba(X_seq)[0]
        else:
            raise ValueError("Unknown model type")

        # Add to prediction buffer for smoothing
        self.prediction_buffer.append(proba)
        if len(self.prediction_buffer) > self.smoothing_window:
            self.prediction_buffer.pop(0)

        # Smoothed prediction
        smoothed_proba = np.mean(self.prediction_buffer)
        is_cheating = smoothed_proba >= self.confidence_threshold

        # Determine alert level
        if smoothed_proba >= 0.9:
            alert_level = "HIGH"
        elif smoothed_proba >= 0.7:
            alert_level = "MEDIUM"
        elif smoothed_proba >= 0.5:
            alert_level = "LOW"
        else:
            alert_level = "NONE"

        result = {
            'is_cheating': is_cheating,
            'confidence': smoothed_proba,
            'raw_confidence': proba,
            'alert_level': alert_level,
            'timestamp': datetime.now().isoformat()
        }

        return result

    def reset(self):
        """Clear all buffers."""
        self.frame_buffer = []
        self.prediction_buffer = []