# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, TimeSeriesSplit
#
# from app.prediction.detectors import XGBoostCheatingDetector
# from app.prediction.utils import DataPreprocessor, FeatureEngineer, evaluate_model, plot_feature_importance
#
#
# def train_xgboost_model(data_path: str, window_size: int = 50):
#     """
#     Complete pipeline for training XGBoost model.
#
#     Args:
#         data_path: Path to the CSV/TSV file with proctoring data
#         window_size: Number of frames to aggregate into windows
#     """
#     print("\n" + "=" * 80)
#     print("TRAINING XGBOOST MODEL")
#     print("=" * 80 + "\n")
#
#     # Step 1: Load and preprocess data
#     print("Step 1: Loading and preprocessing data...")
#     preprocessor = DataPreprocessor()
#     df = preprocessor.load_data(data_path)
#     df = preprocessor.clean_data(df)
#     df = preprocessor.add_student_session_id(df)
#
#     print(f"Data shape: {df.shape}")
#     print(f"Label distribution:\n{df['label'].value_counts()}")
#
#     # Step 2: Feature engineering
#     print("\nStep 2: Creating features...")
#     feature_engineer = FeatureEngineer(window_size=window_size)
#
#     # Add temporal and derived features
#     df = feature_engineer.create_temporal_features(df)
#     df = feature_engineer.create_derived_features(df)
#
#     # Create window-level features (this aggregates frames into windows)
#     window_df = feature_engineer.create_window_features(df, window_size=window_size)
#
#     print("CHECKING VARIOUS THRESHOLDS")
#     for threshold in [0.5, 0.6, 0.7, 0.8]:
#         window_df = feature_engineer.create_window_features(df, label_threshold=threshold)
#         print(f"Threshold {threshold}: {window_df['label'].value_counts()}")
#
#     print(f"Created {len(window_df)} windows from {len(df)} frames")
#     print(f"Window features shape: {window_df.shape}")
#     print(f"Window label distribution:\n{window_df['label'].value_counts()}")
#
#     # Step 3: Split data
#     print("\nStep 3: Splitting data...")
#
#     # Prepare features
#     detector = XGBoostCheatingDetector(window_size=window_size)
#     X, y = detector.prepare_features(window_df)
#
#     # Split by sessions to avoid data leakage
#     unique_sessions = window_df['session_id'].unique()
#     train_sessions, test_sessions = train_test_split(
#         unique_sessions, test_size=0.2, random_state=42
#     )
#     train_sessions, val_sessions = train_test_split(
#         train_sessions, test_size=0.15, random_state=42
#     )
#
#     train_mask = window_df['session_id'].isin(train_sessions)
#     val_mask = window_df['session_id'].isin(val_sessions)
#     test_mask = window_df['session_id'].isin(test_sessions)
#
#     X_train, y_train = X[train_mask], y[train_mask]
#     X_val, y_val = X[val_mask], y[val_mask]
#     X_test, y_test = X[test_mask], y[test_mask]
#
#     print(f"Training set: {len(X_train)} samples")
#     print(f"Validation set: {len(X_val)} samples")
#     print(f"Test set: {len(X_test)} samples")
#
#     # Step 4: Train model
#     print("\nStep 4: Training model...")
#     detector.train(X_train, y_train, X_val, y_val, use_smote=True)
#
#     # Step 5: Evaluate
#     print("\nStep 5: Evaluating model...")
#
#     # Training set performance
#     y_train_pred = detector.predict(X_train)
#     y_train_proba = detector.predict_proba(X_train)
#     print("\n--- Training Set Performance ---")
#     evaluate_model(y_train, y_train_pred, y_train_proba, "XGBoost (Train)")
#
#     # Validation set performance
#     y_val_pred = detector.predict(X_val)
#     y_val_proba = detector.predict_proba(X_val)
#     print("\n--- Validation Set Performance ---")
#     evaluate_model(y_val, y_val_pred, y_val_proba, "XGBoost (Validation)")
#
#     # Test set performance
#     y_test_pred = detector.predict(X_test)
#     y_test_proba = detector.predict_proba(X_test)
#     print("\n--- Test Set Performance ---")
#     test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba, "XGBoost (Test)")
#
#     # Step 6: Feature importance
#     print("\nStep 6: Analyzing feature importance...")
#     feature_importance = detector.get_feature_importance(top_n=20)
#     print("\nTop 20 Most Important Features:")
#     print(feature_importance.to_string(index=False))
#
#     # Plot feature importance
#     plot_feature_importance(feature_importance, top_n=20,
#                             save_path='models/xgboost_feature_importance.png')
#
#     # Step 7: SHAP analysis (if available)
#     print("\nStep 7: Generating SHAP explanations...")
#     try:
#         explainer, shap_values, X_sample = detector.explain_predictions(X_test, num_samples=100)
#         if explainer is not None:
#             print("SHAP analysis completed. You can visualize with shap.summary_plot()")
#     except Exception as e:
#         print(f"SHAP analysis skipped: {e}")
#
#     # Step 8: Save model
#     print("\nStep 8: Saving model...")
#     detector.save('models/xgboost_cheating_detector.pkl')
#
#     print("\n" + "=" * 80)
#     print("XGBOOST TRAINING COMPLETED SUCCESSFULLY!")
#     print("=" * 80 + "\n")
#
#     return detector, test_metrics
#
#
