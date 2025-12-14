"""
Image utility functions for processing and validation.
"""
import base64
import numpy as np
import cv2
from typing import Tuple, Optional
import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


def decode_base64_image(image_base64: str) -> np.ndarray:
    """
    Decode base64-encoded image to numpy array.
    
    Args:
        image_base64: Base64-encoded image string (with or without data URL prefix)
        
    Returns:
        Image as numpy array (BGR format for OpenCV)
    """
    try:
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        return img
        
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise ValueError(f"Invalid image data: {e}")


def encode_image_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """
    Encode numpy image array to base64 string.
    
    Args:
        image: Image as numpy array (BGR format)
        format: Image format ("JPEG" or "PNG")
        
    Returns:
        Base64-encoded image string
    """
    try:
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Encode to bytes
        buffer = BytesIO()
        pil_image.save(buffer, format=format)
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        return f"data:image/{format.lower()};base64,{image_base64}"
        
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        raise


def validate_image_quality(
    image: np.ndarray,
    min_size: Tuple[int, int] = (80, 80),
    min_sharpness: float = 100.0,
    brightness_range: Tuple[int, int] = (40, 220)
) -> Tuple[bool, Optional[str]]:
    """
    Validate image quality for enrollment.
    
    Args:
        image: Input image
        min_size: Minimum image dimensions (width, height)
        min_sharpness: Minimum Laplacian variance
        brightness_range: Acceptable brightness range (min, max)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    h, w = image.shape[:2]
    
    # Check size
    if w < min_size[0] or h < min_size[1]:
        return False, f"Image too small: {w}x{h}, minimum {min_size[0]}x{min_size[1]}"
    
    # Check sharpness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < min_sharpness:
        return False, f"Image too blurry: sharpness {laplacian_var:.2f}, minimum {min_sharpness}"
    
    # Check brightness
    mean_brightness = np.mean(gray)
    if mean_brightness < brightness_range[0] or mean_brightness > brightness_range[1]:
        return False, f"Image brightness out of range: {mean_brightness:.2f}, acceptable {brightness_range[0]}-{brightness_range[1]}"
    
    return True, None

