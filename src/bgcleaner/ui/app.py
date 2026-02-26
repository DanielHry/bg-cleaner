"""BG-cleaner — Streamlit application entry point.

Launch with::

    uv run streamlit run src/bgcleaner/ui/app.py
"""

from __future__ import annotations

import streamlit as st

from bgcleaner.config import Settings
from bgcleaner.core.image_processing import adjust_alpha, composite
from bgcleaner.core.providers import MattingModel, ModelCard, create_engine, discover_available
from bgcleaner.errors import BGCleanerError
from bgcleaner.schemas import CompositeInput, MattingInput
from bgcleaner.ui.state import (
    StateKey,
    clear_pipeline_state,
    get_state,
    has_alpha,
    set_state,
)
from bgcleaner.ui.widgets import (
    render_alpha_preview,
    render_comparison,
    render_downloads,
    render_model_selector,
    render_sidebar_controls,
    render_sidebar_preview,
    render_uploader,
)

# ---------------------------------------------------------------------------
# Page configuration (must be called first)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BG-cleaner",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model…")
def _load_engine(model_id: str, _settings: Settings) -> MattingModel:
    """Load a matting engine, run a warmup inference, and cache it.

    The first ONNX GPU inference triggers CUDA/cuDNN initialisation
    which can take several seconds.  A dummy pass absorbs this cost
    so that the first real image feels responsive.

    Args:
        model_id: Unique identifier of the model to load.
        _settings: Application settings (underscore-prefixed to exclude
            from Streamlit's hash).

    Returns:
        A ``MattingModel`` ready for inference.
    """
    cards = discover_available(_settings)
    card = next(c for c in cards if c.id == model_id)
    engine = create_engine(card, _settings)

    # Warmup: small dummy image to trigger CUDA/cuDNN init.
    from PIL import Image as _Image

    dummy = _Image.new("RGB", (64, 64), (128, 128, 128))
    engine.predict(MattingInput(image=dummy))

    return engine


@st.cache_data(show_spinner=False)
def _get_settings() -> Settings:
    """Load and cache application settings."""
    return Settings()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the BG-cleaner Streamlit application."""
    settings = _get_settings()

    # ---- Header ----
    st.title("✂️ BG-cleaner")
    st.caption("Portrait background removal powered by MODNet & RMBG-2.0.")

    # ---- Sidebar: model selector ----
    available_cards = discover_available(settings)
    selected_card = render_model_selector(available_cards)

    if selected_card is None:
        st.warning(
            f"No ONNX models found in `{settings.models_dir}`. "
            "See the README for download instructions."
        )
        return

    # Clear pipeline if the model changed.
    prev_model_id = get_state(StateKey.SELECTED_MODEL_ID)
    if prev_model_id != selected_card.id:
        clear_pipeline_state()
        set_state(StateKey.SELECTED_MODEL_ID, selected_card.id)

    # ---- Sidebar: alpha controls ----
    params = render_sidebar_controls(settings)

    # ---- Upload ----
    image = render_uploader(settings)

    if image is None:
        st.info("Upload a portrait photo to get started.", icon="📷")
        clear_pipeline_state()
        return

    set_state(StateKey.UPLOADED_IMAGE, image)

    # ---- Sidebar: original preview (small) ----
    render_sidebar_preview(image)

    # ---- Inference ----
    if not has_alpha():
        try:
            engine = _load_engine(selected_card.id, settings)
            with st.spinner("Removing background…"):
                matting_output = engine.predict(MattingInput(image=image))
            set_state(StateKey.MATTING_OUTPUT, matting_output)
        except BGCleanerError as exc:
            st.error(f"Matting failed: {exc}")
            return

    matting_output = get_state(StateKey.MATTING_OUTPUT)

    # ---- Alpha adjustment ----
    adjusted_alpha = adjust_alpha(matting_output.alpha, params)
    set_state(StateKey.ADJUSTED_ALPHA, adjusted_alpha)

    # ---- Compositing ----
    composite_output = composite(
        CompositeInput(image=image, alpha=adjusted_alpha)
    )
    set_state(StateKey.COMPOSITE_OUTPUT, composite_output)

    # ---- Display: comparison slider (original vs result) ----
    st.divider()
    render_comparison(image, composite_output)

    # ---- Display: alpha matte (full width) ----
    st.divider()
    render_alpha_preview(adjusted_alpha)

    # ---- Downloads ----
    st.divider()
    render_downloads(composite_output)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()