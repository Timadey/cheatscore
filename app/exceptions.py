"""
Custom exceptions for the proctoring service.
"""


class EnrollmentError(Exception):
    """Base exception for enrollment errors."""
    pass


class ImageDecodeError(EnrollmentError):
    """Raised when image decoding fails."""
    pass


class ImageQualityError(EnrollmentError):
    """Raised when image quality checks fail."""
    pass


class FaceDetectionError(EnrollmentError):
    """Raised when face detection fails."""
    pass


class MultipleFacesError(EnrollmentError):
    """Raised when multiple faces are detected in an image."""
    pass


class NoFaceDetectedError(EnrollmentError):
    """Raised when no face is detected in an image."""
    pass


class NoValidFacesError(EnrollmentError):
    """Raised when no valid faces are detected in any provided images."""
    pass


class EmbeddingExtractionError(EnrollmentError):
    """Raised when embedding extraction fails."""
    pass


class VerificationError(Exception):
    """Base exception for verification errors."""
    pass


class NoEnrollmentError(VerificationError):
    """Raised when no enrollment is found for a candidate."""
    pass

