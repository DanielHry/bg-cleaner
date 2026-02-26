"""Streamlit session state management.

Centralises all ``st.session_state`` keys and provides typed accessors
so that the rest of the UI layer never uses raw string keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import streamlit as st


class StateKey(str, Enum):
    """All session state keys used by the application."""

    SELECTED_MODEL_ID = "selected_model_id"
    UPLOADED_IMAGE = "uploaded_image"
    MATTING_OUTPUT = "matting_output"
    ADJUSTED_ALPHA = "adjusted_alpha"
    COMPOSITE_OUTPUT = "composite_output"


@dataclass(frozen=True)
class SliderDefaults:
    """Default values for alpha adjustment sliders."""

    brightness: float = 1.0
    contrast: float = 1.0
    sharpness: float = 1.0
    blur_radius: float = 0.0
    threshold_enabled: bool = False
    threshold_value: int = 128


SLIDER_DEFAULTS = SliderDefaults()


def get_state(key: StateKey, default=None):
    """Retrieve a value from session state.

    Args:
        key: The state key to look up.
        default: Fallback value if the key is absent.

    Returns:
        The stored value or *default*.
    """
    return st.session_state.get(key.value, default)


def set_state(key: StateKey, value) -> None:
    """Store a value in session state.

    Args:
        key: The state key to write.
        value: The value to store.
    """
    st.session_state[key.value] = value


def clear_pipeline_state() -> None:
    """Remove all pipeline outputs from session state.

    Called when a new image is uploaded or model is changed so that
    stale results do not leak into the next run.
    """
    for key in (
        StateKey.MATTING_OUTPUT,
        StateKey.ADJUSTED_ALPHA,
        StateKey.COMPOSITE_OUTPUT,
    ):
        st.session_state.pop(key.value, None)


def has_image() -> bool:
    """Return ``True`` if an image has been uploaded."""
    return get_state(StateKey.UPLOADED_IMAGE) is not None


def has_alpha() -> bool:
    """Return ``True`` if the matting output is available."""
    return get_state(StateKey.MATTING_OUTPUT) is not None