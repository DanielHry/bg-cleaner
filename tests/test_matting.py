"""Tests for bgcleaner.core.matting.

The ONNX model is not available in the test environment, so inference
is tested through a mocked ``ort.InferenceSession``.  Preprocessing and
postprocessing logic is exercised directly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from bgcleaner.config import Settings
from bgcleaner.core.matting import MattingEngine, _round_to
from bgcleaner.errors import ModelInferenceError, ModelLoadError
from bgcleaner.schemas import MattingInput


# ---------------------------------------------------------------------------
# _round_to helper
# ---------------------------------------------------------------------------


class TestRoundTo:
    """Validate the rounding helper used in preprocessing."""

    def test_exact_multiple(self) -> None:
        assert _round_to(64, 32) == 64

    def test_rounds_up(self) -> None:
        assert _round_to(50, 32) == 64

    def test_rounds_down(self) -> None:
        assert _round_to(33, 32) == 32

    def test_minimum_is_one_multiple(self) -> None:
        assert _round_to(1, 32) == 32

    def test_zero_returns_multiple(self) -> None:
        assert _round_to(0, 32) == 32

    def test_large_value(self) -> None:
        assert _round_to(1000, 32) == 992


# ---------------------------------------------------------------------------
# Mocked MattingEngine
# ---------------------------------------------------------------------------


def _make_mock_session(output_h: int = 512, output_w: int = 512) -> MagicMock:
    """Create a mock ONNX InferenceSession that returns a dummy alpha.

    Args:
        output_h: Height of the fake model output.
        output_w: Width of the fake model output.

    Returns:
        A configured ``MagicMock`` mimicking ``ort.InferenceSession``.
    """
    session = MagicMock()

    mock_input = MagicMock()
    mock_input.name = "input"
    session.get_inputs.return_value = [mock_input]

    def fake_run(_output_names, input_dict):
        tensor = list(input_dict.values())[0]
        _, _, h, w = tensor.shape
        # Return a gradient alpha: left=0 → right=1.
        alpha = np.linspace(0, 1, w, dtype=np.float32)
        alpha = np.tile(alpha, (1, 1, h, 1))  # (1, 1, H, W)
        return [alpha]

    session.run.side_effect = fake_run
    return session


def _build_engine(mock_session: MagicMock) -> MattingEngine:
    """Construct a MattingEngine with a mocked ONNX session.

    Args:
        mock_session: The mock to inject.

    Returns:
        A ready-to-use ``MattingEngine`` instance.
    """
    with patch("bgcleaner.core.matting.ort.InferenceSession", return_value=mock_session):
        settings = Settings(model_path=Path("/fake/modnet.onnx"))
        # Patch is_file so the path check passes.
        with patch.object(Path, "is_file", return_value=True):
            engine = MattingEngine(settings=settings)
    return engine


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestMattingEngineInit:
    """Validate engine construction and error handling."""

    def test_missing_model_raises_load_error(self) -> None:
        settings = Settings(model_path=Path("/nonexistent/model.onnx"))
        with pytest.raises(ModelLoadError, match="not found"):
            MattingEngine(settings=settings)

    def test_corrupt_model_raises_load_error(self, tmp_path: Path) -> None:
        fake_model = tmp_path / "bad.onnx"
        fake_model.write_bytes(b"not a valid onnx model")
        settings = Settings(model_path=fake_model)
        with pytest.raises(ModelLoadError, match="Failed to load"):
            MattingEngine(settings=settings)


class TestPreprocessing:
    """Validate image preprocessing logic."""

    def test_output_shape_is_nchw(self, rgb_image: Image.Image) -> None:
        mock_session = _make_mock_session()
        engine = _build_engine(mock_session)
        tensor = engine._preprocess(rgb_image)

        assert tensor.ndim == 4
        assert tensor.shape[0] == 1  # batch
        assert tensor.shape[1] == 3  # channels

    def test_output_dtype_is_float32(self, rgb_image: Image.Image) -> None:
        mock_session = _make_mock_session()
        engine = _build_engine(mock_session)
        tensor = engine._preprocess(rgb_image)
        assert tensor.dtype == np.float32

    def test_values_in_minus1_plus1(self, rgb_image: Image.Image) -> None:
        mock_session = _make_mock_session()
        engine = _build_engine(mock_session)
        tensor = engine._preprocess(rgb_image)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_dimensions_are_multiples_of_32(
        self, rgb_image: Image.Image
    ) -> None:
        mock_session = _make_mock_session()
        engine = _build_engine(mock_session)
        tensor = engine._preprocess(rgb_image)
        _, _, h, w = tensor.shape
        assert h % 32 == 0
        assert w % 32 == 0

    def test_converts_rgba_to_rgb(self, rgba_image: Image.Image) -> None:
        """RGBA input should be silently converted to 3-channel RGB."""
        mock_session = _make_mock_session()
        engine = _build_engine(mock_session)
        tensor = engine._preprocess(rgba_image)
        assert tensor.shape[1] == 3

    def test_landscape_and_portrait_scaling(self) -> None:
        """Verify the shortest edge is scaled to ref_size."""
        mock_session = _make_mock_session()
        engine = _build_engine(mock_session)

        landscape = Image.new("RGB", (200, 100))
        tensor = engine._preprocess(landscape)
        _, _, h, w = tensor.shape
        assert min(h, w) == 512  # ref_size default

        portrait = Image.new("RGB", (100, 200))
        tensor = engine._preprocess(portrait)
        _, _, h, w = tensor.shape
        assert min(h, w) == 512


class TestPostprocessing:
    """Validate raw model output → PIL alpha conversion."""

    def test_output_is_pil_mode_l(self) -> None:
        raw = np.random.rand(1, 1, 64, 64).astype(np.float32)
        result = MattingEngine._postprocess(raw, (100, 80))
        assert isinstance(result, Image.Image)
        assert result.mode == "L"

    def test_output_size_matches_original(self) -> None:
        raw = np.random.rand(1, 1, 64, 64).astype(np.float32)
        result = MattingEngine._postprocess(raw, (200, 150))
        assert result.size == (200, 150)

    def test_values_clamped_to_valid_range(self) -> None:
        """Values outside [0, 1] in model output should be clamped."""
        raw = np.array([[[[-0.5, 1.5], [0.0, 1.0]]]], dtype=np.float32)
        result = MattingEngine._postprocess(raw, (2, 2))
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255


class TestPredict:
    """End-to-end predict with mocked inference."""

    def test_predict_returns_matting_output(
        self, rgb_image: Image.Image
    ) -> None:
        mock_session = _make_mock_session()
        engine = _build_engine(mock_session)

        result = engine.predict(MattingInput(image=rgb_image))
        assert result.alpha.mode == "L"
        assert result.alpha.size == rgb_image.size
        assert result.original_size == rgb_image.size

    def test_predict_alpha_has_valid_values(
        self, rgb_image: Image.Image
    ) -> None:
        mock_session = _make_mock_session()
        engine = _build_engine(mock_session)

        result = engine.predict(MattingInput(image=rgb_image))
        arr = np.array(result.alpha)
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_inference_error_is_wrapped(
        self, rgb_image: Image.Image
    ) -> None:
        mock_session = _make_mock_session()
        mock_session.run.side_effect = RuntimeError("ONNX Runtime boom")
        engine = _build_engine(mock_session)

        with pytest.raises(ModelInferenceError, match="boom"):
            engine.predict(MattingInput(image=rgb_image))