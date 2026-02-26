"""Tests for bgcleaner.schemas."""

import pytest
from PIL import Image

from bgcleaner.schemas import (
    AlphaAdjustmentParams,
    CompositeInput,
    CompositeOutput,
    MattingInput,
    MattingOutput,
)


# ---------------------------------------------------------------------------
# MattingInput
# ---------------------------------------------------------------------------


class TestMattingInput:
    """Validate MattingInput accepts PIL images."""

    def test_accepts_rgb_image(self, rgb_image: Image.Image) -> None:
        mi = MattingInput(image=rgb_image)
        assert mi.image.mode == "RGB"

    def test_accepts_rgba_image(self, rgba_image: Image.Image) -> None:
        mi = MattingInput(image=rgba_image)
        assert mi.image.mode == "RGBA"

    def test_rejects_non_image(self) -> None:
        with pytest.raises(Exception):
            MattingInput(image="not_an_image")


# ---------------------------------------------------------------------------
# MattingOutput
# ---------------------------------------------------------------------------


class TestMattingOutput:
    """Validate MattingOutput structure."""

    def test_construction(self, alpha_image: Image.Image) -> None:
        mo = MattingOutput(alpha=alpha_image, original_size=(100, 80))
        assert mo.alpha.mode == "L"
        assert mo.original_size == (100, 80)


# ---------------------------------------------------------------------------
# AlphaAdjustmentParams
# ---------------------------------------------------------------------------


class TestAlphaAdjustmentParams:
    """Validate defaults, boundaries, and the has_adjustments helper."""

    def test_defaults_are_neutral(self) -> None:
        params = AlphaAdjustmentParams()
        assert params.brightness == 1.0
        assert params.contrast == 1.0
        assert params.sharpness == 1.0
        assert params.blur_radius == 0.0
        assert params.threshold is None

    def test_has_adjustments_false_on_defaults(self) -> None:
        assert AlphaAdjustmentParams().has_adjustments() is False

    def test_has_adjustments_true_on_brightness(self) -> None:
        assert AlphaAdjustmentParams(brightness=1.5).has_adjustments() is True

    def test_has_adjustments_true_on_contrast(self) -> None:
        assert AlphaAdjustmentParams(contrast=0.8).has_adjustments() is True

    def test_has_adjustments_true_on_sharpness(self) -> None:
        assert AlphaAdjustmentParams(sharpness=2.0).has_adjustments() is True

    def test_has_adjustments_true_on_blur(self) -> None:
        assert AlphaAdjustmentParams(blur_radius=1.0).has_adjustments() is True

    def test_has_adjustments_true_on_threshold(self) -> None:
        assert AlphaAdjustmentParams(threshold=128).has_adjustments() is True

    def test_brightness_lower_bound(self) -> None:
        with pytest.raises(Exception):
            AlphaAdjustmentParams(brightness=-0.1)

    def test_brightness_upper_bound(self) -> None:
        with pytest.raises(Exception):
            AlphaAdjustmentParams(brightness=3.1)

    def test_threshold_lower_bound(self) -> None:
        with pytest.raises(Exception):
            AlphaAdjustmentParams(threshold=-1)

    def test_threshold_upper_bound(self) -> None:
        with pytest.raises(Exception):
            AlphaAdjustmentParams(threshold=256)

    def test_threshold_accepts_zero(self) -> None:
        params = AlphaAdjustmentParams(threshold=0)
        assert params.threshold == 0

    def test_threshold_accepts_255(self) -> None:
        params = AlphaAdjustmentParams(threshold=255)
        assert params.threshold == 255

    def test_blur_rejects_negative(self) -> None:
        with pytest.raises(Exception):
            AlphaAdjustmentParams(blur_radius=-1.0)


# ---------------------------------------------------------------------------
# CompositeInput / CompositeOutput
# ---------------------------------------------------------------------------


class TestCompositeSchemas:
    """Validate compositing data contracts."""

    def test_composite_input(
        self, rgb_image: Image.Image, alpha_image: Image.Image
    ) -> None:
        ci = CompositeInput(image=rgb_image, alpha=alpha_image)
        assert ci.image.size == (100, 80)
        assert ci.alpha.size == (100, 80)

    def test_composite_output(
        self, rgba_image: Image.Image, alpha_image: Image.Image
    ) -> None:
        co = CompositeOutput(rgba_image=rgba_image, alpha_image=alpha_image)
        assert co.rgba_image.mode == "RGBA"
        assert co.alpha_image.mode == "L"