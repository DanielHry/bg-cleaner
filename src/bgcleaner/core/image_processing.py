"""Alpha matte adjustment and RGBA compositing.

All functions in this module operate on PIL images and are completely
independent of the inference engine and the UI layer.

Typical usage::

    from bgcleaner.core.image_processing import adjust_alpha, composite
    from bgcleaner.schemas import AlphaAdjustmentParams, CompositeInput

    params = AlphaAdjustmentParams(brightness=1.2, contrast=1.1)
    adjusted = adjust_alpha(alpha_image, params)

    result = composite(CompositeInput(image=original, alpha=adjusted))
    result.rgba_image.save("output.png")
"""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageEnhance, ImageFilter

from bgcleaner.errors import AlphaProcessingError
from bgcleaner.schemas import (
    AlphaAdjustmentParams,
    CompositeInput,
    CompositeOutput,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alpha adjustment
# ---------------------------------------------------------------------------


def adjust_alpha(
    alpha: Image.Image,
    params: AlphaAdjustmentParams,
) -> Image.Image:
    """Apply enhancement filters to an alpha matte.

    Processing order: brightness → contrast → sharpness → blur → threshold.
    If no parameter differs from the neutral defaults the original image is
    returned unchanged (no copy).

    Args:
        alpha: Grayscale PIL image in mode ``L``.
        params: Adjustment parameters.

    Returns:
        A new PIL image in mode ``L`` with the adjustments applied, or the
        original image if ``params.has_adjustments()`` is ``False``.

    Raises:
        AlphaProcessingError: If any PIL operation fails unexpectedly.
    """
    if not params.has_adjustments():
        return alpha

    try:
        result = alpha.convert("L")

        if params.brightness != 1.0:
            result = ImageEnhance.Brightness(result).enhance(params.brightness)

        if params.contrast != 1.0:
            result = ImageEnhance.Contrast(result).enhance(params.contrast)

        if params.sharpness != 1.0:
            result = ImageEnhance.Sharpness(result).enhance(params.sharpness)

        if params.threshold is not None:
            lut = [0 if v <= params.threshold else 255 for v in range(256)]
            result = result.point(lut)

        if params.blur_radius > 0.0:
            result = result.filter(
                ImageFilter.GaussianBlur(radius=params.blur_radius)
            )

        

    except Exception as exc:
        raise AlphaProcessingError(
            f"Alpha adjustment failed: {exc}"
        ) from exc

    return result


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def composite(input_data: CompositeInput) -> CompositeOutput:
    """Compose an RGBA image from the original RGB and an alpha matte.

    If the alpha matte dimensions do not match the original image it is
    resized automatically with bilinear interpolation.

    Args:
        input_data: Validated input containing the original image and its
            alpha matte.

    Returns:
        A ``CompositeOutput`` holding both the RGBA result and the
        standalone alpha matte.
    """
    rgb = input_data.image.convert("RGB")
    alpha = input_data.alpha.convert("L")

    if rgb.size != alpha.size:
        logger.debug(
            "Alpha size %s differs from image size %s — resizing.",
            alpha.size,
            rgb.size,
        )
        alpha = alpha.resize(rgb.size, Image.BILINEAR)

    rgba = rgb.copy()
    rgba.putalpha(alpha)

    return CompositeOutput(rgba_image=rgba, alpha_image=alpha)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Serialize a PIL image to PNG bytes.

    Useful for Streamlit download buttons and API responses.

    Args:
        image: Any PIL image (RGBA, RGB, L, …).

    Returns:
        Raw PNG file contents as ``bytes``.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()