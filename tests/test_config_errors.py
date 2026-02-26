"""Tests for bgcleaner.config and bgcleaner.errors."""

from bgcleaner.config import Settings
from bgcleaner.errors import (
    AlphaProcessingError,
    BGCleanerError,
    InvalidImageError,
    ModelInferenceError,
    ModelLoadError,
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    """Validate default configuration values."""

    def test_default_ref_size(self) -> None:
        s = Settings()
        assert s.ref_size == 512

    def test_default_providers(self) -> None:
        s = Settings()
        assert s.onnx_providers == ["CPUExecutionProvider"]

    def test_default_supported_formats(self) -> None:
        s = Settings()
        assert "jpg" in s.supported_formats
        assert "png" in s.supported_formats

    def test_model_path_ends_with_onnx(self) -> None:
        s = Settings()
        assert s.model_path.suffix == ".onnx"

    def test_slider_ranges_are_valid(self) -> None:
        s = Settings()
        lo, hi = s.alpha_brightness_range
        assert lo < hi
        lo, hi = s.alpha_contrast_range
        assert lo < hi


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    """Verify that all custom exceptions inherit from BGCleanerError."""

    def test_model_load_error(self) -> None:
        assert issubclass(ModelLoadError, BGCleanerError)

    def test_model_inference_error(self) -> None:
        assert issubclass(ModelInferenceError, BGCleanerError)

    def test_invalid_image_error(self) -> None:
        assert issubclass(InvalidImageError, BGCleanerError)

    def test_alpha_processing_error(self) -> None:
        assert issubclass(AlphaProcessingError, BGCleanerError)

    def test_base_inherits_from_exception(self) -> None:
        assert issubclass(BGCleanerError, Exception)

    def test_catch_all_with_base(self) -> None:
        """All specific errors should be catchable via the base class."""
        for exc_class in (
            ModelLoadError,
            ModelInferenceError,
            InvalidImageError,
            AlphaProcessingError,
        ):
            try:
                raise exc_class("test")
            except BGCleanerError:
                pass  # Expected.