"""Tests for bgcleaner.core.image_processing."""

import numpy as np
import pytest
from PIL import Image

from bgcleaner.core.image_processing import (
    adjust_alpha,
    composite,
    image_to_png_bytes,
)
from bgcleaner.schemas import AlphaAdjustmentParams, CompositeInput


# ---------------------------------------------------------------------------
# adjust_alpha
# ---------------------------------------------------------------------------


class TestAdjustAlpha:
    """Validate alpha matte post-processing pipeline."""

    def test_no_adjustment_returns_same_object(
        self, alpha_image: Image.Image
    ) -> None:
        """When all params are neutral, the exact same object is returned."""
        params = AlphaAdjustmentParams()
        result = adjust_alpha(alpha_image, params)
        assert result is alpha_image

    def test_brightness_increase_raises_mean(
        self, uniform_alpha: Image.Image
    ) -> None:
        params = AlphaAdjustmentParams(brightness=2.0)
        result = adjust_alpha(uniform_alpha, params)
        assert np.array(result).mean() > np.array(uniform_alpha).mean()

    def test_brightness_decrease_lowers_mean(
        self, uniform_alpha: Image.Image
    ) -> None:
        params = AlphaAdjustmentParams(brightness=0.5)
        result = adjust_alpha(uniform_alpha, params)
        assert np.array(result).mean() < np.array(uniform_alpha).mean()

    def test_contrast_preserves_mode(self, alpha_image: Image.Image) -> None:
        params = AlphaAdjustmentParams(contrast=1.5)
        result = adjust_alpha(alpha_image, params)
        assert result.mode == "L"
        assert result.size == alpha_image.size

    def test_sharpness_preserves_size(self, alpha_image: Image.Image) -> None:
        params = AlphaAdjustmentParams(sharpness=2.0)
        result = adjust_alpha(alpha_image, params)
        assert result.size == alpha_image.size

    def test_blur_smooths_edges(self, alpha_image: Image.Image) -> None:
        """A blurred alpha should have fewer extreme pixel transitions."""
        params = AlphaAdjustmentParams(blur_radius=3.0)
        result = adjust_alpha(alpha_image, params)

        original_arr = np.array(alpha_image, dtype=np.float32)
        blurred_arr = np.array(result, dtype=np.float32)

        # Standard deviation should decrease after blurring.
        assert blurred_arr.std() < original_arr.std()

    def test_threshold_binarizes(self, uniform_alpha: Image.Image) -> None:
        """A uniform 128 image thresholded at 100 should become all white."""
        params = AlphaAdjustmentParams(threshold=100)
        result = adjust_alpha(uniform_alpha, params)
        arr = np.array(result)
        assert np.all(arr == 255)

    def test_threshold_at_200_binarizes_to_black(
        self, uniform_alpha: Image.Image
    ) -> None:
        """A uniform 128 image thresholded at 200 should become all black."""
        params = AlphaAdjustmentParams(threshold=200)
        result = adjust_alpha(uniform_alpha, params)
        arr = np.array(result)
        assert np.all(arr == 0)

    def test_combined_adjustments(self, alpha_image: Image.Image) -> None:
        """Multiple adjustments can be applied in a single call."""
        params = AlphaAdjustmentParams(
            brightness=1.2,
            contrast=1.3,
            blur_radius=1.0,
        )
        result = adjust_alpha(alpha_image, params)
        assert result.mode == "L"
        assert result.size == alpha_image.size

    def test_output_values_within_valid_range(
        self, alpha_image: Image.Image
    ) -> None:
        params = AlphaAdjustmentParams(brightness=2.5, contrast=2.5)
        result = adjust_alpha(alpha_image, params)
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255


# ---------------------------------------------------------------------------
# composite
# ---------------------------------------------------------------------------


class TestComposite:
    """Validate RGBA compositing."""

    def test_output_mode_is_rgba(
        self, rgb_image: Image.Image, alpha_image: Image.Image
    ) -> None:
        result = composite(CompositeInput(image=rgb_image, alpha=alpha_image))
        assert result.rgba_image.mode == "RGBA"

    def test_output_alpha_mode_is_l(
        self, rgb_image: Image.Image, alpha_image: Image.Image
    ) -> None:
        result = composite(CompositeInput(image=rgb_image, alpha=alpha_image))
        assert result.alpha_image.mode == "L"

    def test_output_size_matches_input(
        self, rgb_image: Image.Image, alpha_image: Image.Image
    ) -> None:
        result = composite(CompositeInput(image=rgb_image, alpha=alpha_image))
        assert result.rgba_image.size == rgb_image.size

    def test_alpha_channel_matches_matte(
        self, rgb_image: Image.Image, alpha_image: Image.Image
    ) -> None:
        """The RGBA alpha channel should equal the input matte."""
        result = composite(CompositeInput(image=rgb_image, alpha=alpha_image))
        rgba_arr = np.array(result.rgba_image)
        alpha_arr = np.array(alpha_image)
        np.testing.assert_array_equal(rgba_arr[:, :, 3], alpha_arr)

    def test_rgb_channels_preserved(
        self, rgb_image: Image.Image, alpha_image: Image.Image
    ) -> None:
        """The RGB channels should be identical to the original image."""
        result = composite(CompositeInput(image=rgb_image, alpha=alpha_image))
        rgba_arr = np.array(result.rgba_image)
        rgb_arr = np.array(rgb_image)
        np.testing.assert_array_equal(rgba_arr[:, :, :3], rgb_arr)

    def test_mismatched_sizes_auto_resize(
        self, rgb_image: Image.Image
    ) -> None:
        """Alpha is resized if it doesn't match the image dimensions."""
        small_alpha = Image.new("L", (50, 40), 200)
        result = composite(
            CompositeInput(image=rgb_image, alpha=small_alpha)
        )
        assert result.rgba_image.size == rgb_image.size
        assert result.alpha_image.size == rgb_image.size

    def test_full_opaque_alpha(self, rgb_image: Image.Image) -> None:
        """With a fully white alpha, all pixels should be fully opaque."""
        white_alpha = Image.new("L", rgb_image.size, 255)
        result = composite(
            CompositeInput(image=rgb_image, alpha=white_alpha)
        )
        arr = np.array(result.rgba_image)
        assert np.all(arr[:, :, 3] == 255)

    def test_full_transparent_alpha(self, rgb_image: Image.Image) -> None:
        """With a fully black alpha, all pixels should be transparent."""
        black_alpha = Image.new("L", rgb_image.size, 0)
        result = composite(
            CompositeInput(image=rgb_image, alpha=black_alpha)
        )
        arr = np.array(result.rgba_image)
        assert np.all(arr[:, :, 3] == 0)


# ---------------------------------------------------------------------------
# image_to_png_bytes
# ---------------------------------------------------------------------------


class TestImageToPngBytes:
    """Validate PNG export helper."""

    def test_returns_bytes(self, rgb_image: Image.Image) -> None:
        data = image_to_png_bytes(rgb_image)
        assert isinstance(data, bytes)

    def test_png_signature(self, rgb_image: Image.Image) -> None:
        """PNG files always start with the 8-byte magic signature."""
        data = image_to_png_bytes(rgb_image)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_roundtrip(self, rgb_image: Image.Image) -> None:
        """Exporting and re-importing should preserve dimensions."""
        import io

        data = image_to_png_bytes(rgb_image)
        reloaded = Image.open(io.BytesIO(data))
        assert reloaded.size == rgb_image.size

    def test_rgba_export(self, rgba_image: Image.Image) -> None:
        data = image_to_png_bytes(rgba_image)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_alpha_export(self, alpha_image: Image.Image) -> None:
        data = image_to_png_bytes(alpha_image)
        assert len(data) > 0