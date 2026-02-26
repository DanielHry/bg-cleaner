"""Custom exceptions for the BG-cleaner pipeline."""


class BGCleanerError(Exception):
    """Base exception for all BG-cleaner errors."""


class ModelLoadError(BGCleanerError):
    """Raised when the ONNX model cannot be loaded.

    Typical causes: missing file, corrupted weights, unsupported opset.
    """


class ModelInferenceError(BGCleanerError):
    """Raised when ONNX inference fails at runtime.

    Typical causes: unexpected input shape, provider error, OOM.
    """


class InvalidImageError(BGCleanerError):
    """Raised when the input image is invalid or unsupported.

    Typical causes: corrupted file, unsupported format, zero-size image.
    """


class AlphaProcessingError(BGCleanerError):
    """Raised when alpha matte post-processing fails."""