"""Pydantic data contracts shared across modules.

Every cross-module boundary is typed through one of these schemas.
The pipeline flow is::

    PIL.Image (upload)
        → MattingInput  → MattingEngine.predict() → MattingOutput
        → adjust_alpha(alpha_image, AlphaAdjustmentParams) → PIL.Image
        → CompositeInput → composite()              → CompositeOutput
"""

from __future__ import annotations

from PIL import Image
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Matting (inference)
# ---------------------------------------------------------------------------


class MattingInput(BaseModel):
    """Input contract for the MODNet inference engine.

    Attributes:
        image: RGB PIL image of arbitrary size.
    """

    image: Image.Image

    model_config = {"arbitrary_types_allowed": True}


class MattingOutput(BaseModel):
    """Output contract returned by the MODNet inference engine.

    Attributes:
        alpha: Grayscale PIL image (mode ``L``) with the same dimensions
            as the original input. Pixel values range from 0 (background)
            to 255 (foreground).
        original_size: Original ``(width, height)`` of the input image,
            preserved so downstream consumers can verify spatial consistency.
    """

    alpha: Image.Image = Field(
        ...,
        description="Alpha matte as a PIL Image in mode 'L'.",
    )
    original_size: tuple[int, int] = Field(
        ...,
        description="Original (width, height) of the input image.",
    )

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Alpha adjustment
# ---------------------------------------------------------------------------


class AlphaAdjustmentParams(BaseModel):
    """Parameters for post-processing the alpha matte.

    All enhancement factors follow the PIL ``ImageEnhance`` convention:
    ``1.0`` means *no change*, values below reduce the effect, values
    above amplify it.

    Attributes:
        brightness: Brightness enhancement factor.
        contrast: Contrast enhancement factor.
        sharpness: Sharpness enhancement factor.
        blur_radius: Gaussian blur radius in pixels. ``0.0`` disables blur.
        threshold: Optional hard-cutoff threshold in ``[0, 255]``. Pixels
            above this value become 255, others become 0. ``None`` means
            no thresholding is applied.
    """

    brightness: float = Field(default=1.0, ge=0.0, le=3.0)
    contrast: float = Field(default=1.0, ge=0.0, le=3.0)
    sharpness: float = Field(default=1.0, ge=0.0, le=3.0)
    blur_radius: float = Field(default=0.0, ge=0.0, le=10.0)
    threshold: int | None = Field(default=None, ge=0, le=255)

    def has_adjustments(self) -> bool:
        """Return ``True`` if any parameter differs from the neutral default."""
        return (
            self.brightness != 1.0
            or self.contrast != 1.0
            or self.sharpness != 1.0
            or self.blur_radius > 0.0
            or self.threshold is not None
        )


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


class CompositeInput(BaseModel):
    """Input contract for RGBA compositing.

    Attributes:
        image: Original RGB image.
        alpha: Alpha matte in mode ``L`` (grayscale), same size as *image*.
    """

    image: Image.Image
    alpha: Image.Image = Field(
        ...,
        description="Alpha matte in PIL mode 'L'.",
    )

    model_config = {"arbitrary_types_allowed": True}


class CompositeOutput(BaseModel):
    """Output contract for RGBA compositing.

    Attributes:
        rgba_image: Final image with transparency applied (mode ``RGBA``).
        alpha_image: Standalone alpha matte (mode ``L``), useful for
            separate download.
    """

    rgba_image: Image.Image = Field(
        ...,
        description="Composited RGBA image.",
    )
    alpha_image: Image.Image = Field(
        ...,
        description="Alpha matte in mode 'L'.",
    )

    model_config = {"arbitrary_types_allowed": True}