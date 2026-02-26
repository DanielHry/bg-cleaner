"""Reusable Streamlit UI components.

Each function renders a self-contained section of the interface.
Business logic is deliberately kept out — widgets call back into
``core`` via the schemas only.
"""

from __future__ import annotations

import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

from bgcleaner.config import Settings
from bgcleaner.core.image_processing import image_to_png_bytes
from bgcleaner.core.providers._base import ModelCard
from bgcleaner.schemas import AlphaAdjustmentParams, CompositeOutput
from bgcleaner.ui.state import SLIDER_DEFAULTS


# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------


def render_model_selector(available_cards: list[ModelCard]) -> ModelCard | None:
    """Render a model selection dropdown in the sidebar.

    Args:
        available_cards: List of model cards found on disk.

    Returns:
        The selected ``ModelCard``, or ``None`` if no models are
        available.
    """
    if not available_cards:
        st.sidebar.error("No models found in the models directory.")
        return None

    card_by_name = {card.name: card for card in available_cards}
    selected_name = st.sidebar.selectbox(
        "Model",
        options=list(card_by_name.keys()),
        help="Choose a matting model. Only models present in assets/models/ are listed.",
    )
    return card_by_name[selected_name]


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def render_uploader(settings: Settings) -> Image.Image | None:
    """Render the image upload widget.

    Args:
        settings: Application settings (used for allowed formats and
            max file size).

    Returns:
        A PIL image if a file was uploaded, otherwise ``None``.
    """
    uploaded = st.file_uploader(
        "Upload a portrait photo",
        type=settings.supported_formats,
        help=f"Max {settings.max_upload_mb:.0f} MB — JPG, PNG, or WebP.",
    )
    if uploaded is None:
        return None
    return Image.open(uploaded)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


def render_sidebar_controls(settings: Settings) -> AlphaAdjustmentParams:
    """Render the alpha adjustment sliders in the sidebar.

    Processing order: brightness → contrast → sharpness → threshold → blur.

    Args:
        settings: Application settings (used for slider ranges).

    Returns:
        An ``AlphaAdjustmentParams`` populated from the current slider
        values.
    """
    st.sidebar.header("Alpha adjustments")

    b_lo, b_hi = settings.alpha_brightness_range
    brightness = st.sidebar.slider(
        "Brightness",
        min_value=b_lo,
        max_value=b_hi,
        value=SLIDER_DEFAULTS.brightness,
        step=0.05,
        help="1.0 = no change.",
    )

    c_lo, c_hi = settings.alpha_contrast_range
    contrast = st.sidebar.slider(
        "Contrast",
        min_value=c_lo,
        max_value=c_hi,
        value=SLIDER_DEFAULTS.contrast,
        step=0.05,
        help="1.0 = no change.",
    )

    s_lo, s_hi = settings.alpha_sharpness_range
    sharpness = st.sidebar.slider(
        "Sharpness",
        min_value=s_lo,
        max_value=s_hi,
        value=SLIDER_DEFAULTS.sharpness,
        step=0.05,
        help="1.0 = no change.",
    )

    # Threshold (opt-in).
    threshold: int | None = None
    use_threshold = st.sidebar.checkbox(
        "Enable threshold",
        value=SLIDER_DEFAULTS.threshold_enabled,
        help="Binarise the alpha matte at a hard cutoff.",
    )
    if use_threshold:
        threshold = st.sidebar.slider(
            "Threshold",
            min_value=0,
            max_value=255,
            value=SLIDER_DEFAULTS.threshold_value,
            step=1,
        )

    # Blur (after threshold so it smooths binarised edges).
    blur_radius = st.sidebar.slider(
        "Edge blur",
        min_value=0.0,
        max_value=settings.alpha_blur_max,
        value=SLIDER_DEFAULTS.blur_radius,
        step=0.5,
        help="Gaussian blur radius to soften matte edges.",
    )

    return AlphaAdjustmentParams(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        blur_radius=blur_radius,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Sidebar original preview
# ---------------------------------------------------------------------------


def render_sidebar_preview(image: Image.Image) -> None:
    """Display a small preview of the original image in the sidebar.

    Args:
        image: The uploaded PIL image.
    """
    st.sidebar.divider()
    st.sidebar.image(image, caption="Original", use_container_width=True)


# ---------------------------------------------------------------------------
# Main result area
# ---------------------------------------------------------------------------


def render_comparison(
    original: Image.Image,
    composite_output: CompositeOutput,
) -> None:
    """Render a before/after slider comparing original and result.

    Uses ``streamlit-image-comparison`` for an interactive overlay,
    centred on the page regardless of the Streamlit layout mode.

    Args:
        original: The uploaded original image.
        composite_output: The compositing result containing RGBA.
    """
    # Convert RGBA result to RGB with white background for the slider.
    rgba = composite_output.rgba_image
    white_bg = Image.new("RGB", rgba.size, (255, 255, 255))
    white_bg.paste(rgba, mask=rgba.split()[3])

    # Centre the component using columns.
    _, center, _ = st.columns([1, 3, 1])
    with center:
        image_comparison(
            img1=original.convert("RGB"),
            img2=white_bg,
            label1="Original",
            label2="Background removed",
            width=1024,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True,
        )


def render_alpha_preview(alpha: Image.Image) -> None:
    """Display the alpha matte centred and matching the comparison width.

    Args:
        alpha: The (possibly adjusted) alpha matte in mode ``L``.
    """
    _, center, _ = st.columns([1, 3, 1])
    with center:
        st.image(alpha, caption="Alpha matte", use_container_width=True)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def render_downloads(composite_output: CompositeOutput) -> None:
    """Render download buttons for the RGBA image and the alpha mask.

    Args:
        composite_output: The compositing result to export.
    """
    _, center, _ = st.columns([1, 3, 1])
    with center:
        col_dl_rgba, col_dl_alpha = st.columns(2)

        with col_dl_rgba:
            st.download_button(
                label="⬇ Download PNG (transparent)",
                data=image_to_png_bytes(composite_output.rgba_image),
                file_name="bg_removed.png",
                mime="image/png",
                use_container_width=True,
            )

        with col_dl_alpha:
            st.download_button(
                label="⬇ Download alpha mask",
                data=image_to_png_bytes(composite_output.alpha_image),
                file_name="alpha_mask.png",
                mime="image/png",
                use_container_width=True,
            )