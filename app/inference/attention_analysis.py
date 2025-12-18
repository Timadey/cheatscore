"""
Attention analysis module using MediaPipe Face Mesh.
estimates head pose and gaze direction.
"""
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class GazeDirection(str, Enum):
    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class AttentionResult:
    def __init__(
        self,
        gaze_direction: GazeDirection,
        head_pose: Dict[str, float],
        is_looking_at_screen: bool,
        landmarks: Optional[Any] = None
    ):
        self.gaze_direction = gaze_direction
        self.head_pose = head_pose
        self.is_looking_at_screen = is_looking_at_screen
        self.landmarks = landmarks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gaze_direction": self.gaze_direction.value,
            "head_pose": self.head_pose,
            "is_looking_at_screen": self.is_looking_at_screen
        }


class AttentionAnalyzer:
    """
    Analyzes visual attention using Gaze Estimation and Head Pose.
    Uses MediaPipe Face Mesh.
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Crucial for iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 3D Model Points for PnP (Generic face model)
        # Nose tip, Chin, Left Eye Left Corner, Right Eye Right Corner, Left Mouth Corner, Right Mouth Corner
        self.face_3d = np.array([
            [0.0, 0.0, 0.0],            # Nose tip
            [0.0, -330.0, -65.0],       # Chin
            [-225.0, 170.0, -135.0],    # Left eye left corner
            [225.0, 170.0, -135.0],     # Right eye right corner
            [-150.0, -150.0, -125.0],   # Left Mouth corner
            [150.0, -150.0, -125.0]     # Right mouth corner
        ], dtype=np.float64)

        # Keypoint indices in MediaPipe Face Mesh (Canonical)
        # 1: Nose tip, 152: Chin, 33: Left eye inner, 263: Right eye inner, 61: Mouth left, 291: Mouth right
        # Using corners for better stability with PnP: 
        # 1: Nose tip
        # 199: Chin
        # 33: Left eye left corner (approx)
        # 263: Right eye right corner (approx)
        # 61: Left mouth corner
        # 291: Right mouth corner
        self.keypoint_indices = [1, 199, 33, 263, 61, 291]

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()

    def process_frame(self, frame: np.ndarray) -> Optional[AttentionResult]:
        """
        Process a single frame to extract attention metrics.
        Frame should be BGR (OpenCV standard).
        """
        try:
            h, w, _ = frame.shape
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            results = self.face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                return None
            
            # We assume single face for attention analysis (proctoring focus)
            face_landmarks = results.multi_face_landmarks[0]
            
            # 1. Estimate Head Pose
            pitch, yaw, roll = self._estimate_head_pose(face_landmarks, w, h)
            
            # 2. Estimate Gaze
            gaze_dir, is_centered_gaze = self._estimate_gaze(face_landmarks, w, h, pitch, yaw)
            
            # 3. Determine overall "Looking at Screen" status
            # Check if head pose is within acceptable range
            is_pose_acceptable = (
                abs(pitch) < settings.head_pose_pitch_threshold and
                abs(yaw) < settings.head_pose_yaw_threshold
            )
            
            is_looking_at_screen = is_pose_acceptable and is_centered_gaze
            
            return AttentionResult(
                gaze_direction=gaze_dir,
                head_pose={"pitch": pitch, "yaw": yaw, "roll": roll},
                is_looking_at_screen=is_looking_at_screen,
                landmarks=face_landmarks
            )

        except Exception as e:
            logger.error(f"Attention analysis failed: {e}")
            return None

    def _estimate_head_pose(self, landmarks, img_w, img_h) -> Tuple[float, float, float]:
        """
        Estimate head pose (Pitch, Yaw, Roll) using SolvePnP.
        """
        face_2d = []
        for idx in self.keypoint_indices:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            
        face_2d = np.array(face_2d, dtype=np.float64)

        # Camera matrix (Assume focal length = width)
        focal_length = 1.0 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ])
        
        # Dist coeff (Assume no lens distortion)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(
            self.face_3d, face_2d, cam_matrix, dist_matrix
        )
        
        # Get rotation matrix
        rmat, jac = cv2.Rodrigues(rot_vec)
        
        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        # Angles are in degrees
        pitch = angles[0] * 360
        yaw = angles[1] * 360
        roll = angles[2] * 360
        
        return pitch, yaw, roll

    def _estimate_gaze(
        self, 
        landmarks, 
        img_w, 
        img_h, 
        head_pitch: float, 
        head_yaw: float
    ) -> Tuple[GazeDirection, bool]:
        """
        Estimate gaze direction based on Iris position and Head Pose.
        """
        # Iris indices (Left and Right)
        # Left Iris Center: 468, Right Iris Center: 473
        # Left Eye Corners: 33 (inner), 133 (outer) -- wait, 33 is outer in some maps, let's verify
        # Standard MP Mesh:
        # Left Eye: 33 (left/outer), 133 (right/inner)
        # Right Eye: 362 (left/inner), 263 (right/outer)
        
        # Get iris positions
        left_iris = landmarks.landmark[468]
        right_iris = landmarks.landmark[473]
        
        # We can implement a simple heuristic based on relative position in the eye bounding box
        # But robust "looking at screen" often relies heavly on head pose too.
        
        # If head pose is extreme, we trust head pose primarily
        if abs(head_yaw) > 35:
            return (GazeDirection.LEFT if head_yaw < 0 else GazeDirection.RIGHT), False
        if head_pitch > 30: # Look up
             return GazeDirection.UP, False
        if head_pitch < -30: # Look down
             return GazeDirection.DOWN, False

        # If head pose is relatively center, check eyes
        # Determine horizontal ratio
        # normalized_x of iris relative to eye width
        
        # Function to get ratio
        def get_ratio(eye_outer_idx, eye_inner_idx, iris_idx):
            outer = landmarks.landmark[eye_outer_idx]
            inner = landmarks.landmark[eye_inner_idx]
            iris = landmarks.landmark[iris_idx]
            
            # Distance
            total_width = abs(outer.x - inner.x)
            if total_width == 0: return 0.5
            
            # dist from outer
            dist_iris = abs(iris.x - outer.x)
            
            return dist_iris / total_width

        # Left Eye (33 outer left, 133 inner right for left eye? No. 
        # MP Topology: Left eye is on the viewer's left (subject's right) or vice versa?
        # MP 468 is Left Iris (Subject's Left).
        # Subject Left Eye indices: 33 (outer/temporal), 133 (inner/nasal)
        # Ratio 0 = Looking Left (Subject perspective), Ratio 1 = Looking Right
        
        left_ratio = get_ratio(33, 133, 468)
        right_ratio = get_ratio(362, 263, 473)
        
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # Determine Vertical
        # ... Vertical is harder with just aspect ratio, using head pose pitch is safer for "Up/Down" 
        # unless iris is clearly top/bottom.
        
        # Heuristics
        gaze_dir = GazeDirection.CENTER
        is_centered = True
        
        # Horizontal
        if avg_ratio < 0.35: # Looking Left (Subject's right in image if mirrored? No, MP coordinates are normalized image space)
            # x increases left to right.
            # If iris is closer to left x (33), ratio is small.
            # 33 is Left Eye temporal (leftmost point of left eye).
            # So ratio small -> Looking Left (Subject's Right). 
            # Wait, let's stick to Screen Directions.
            # Screen Left (User looks to their Right) -> Image Left.
            gaze_dir = GazeDirection.RIGHT # Subject looks to THEIR right (Screen Left)
            is_centered = False
        elif avg_ratio > 0.65:
            gaze_dir = GazeDirection.LEFT # Subject looks to THEIR left
            is_centered = False
            
        # Vertical from Pose (Dominant factor usually)
        if head_pitch < -20: # Chin down
            gaze_dir = GazeDirection.DOWN
            is_centered = False
        elif head_pitch > 20: # Chin up
            gaze_dir = GazeDirection.UP
            is_centered = False
            
        return gaze_dir, is_centered

