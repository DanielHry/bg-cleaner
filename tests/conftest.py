"""Shared fixtures for BG-cleaner test suite."""

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rgb_image() -> Image.Image:
    """A small 100x80 RGB test image with a gradient."""
    arr = np.zeros((80, 100, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 100, dtype=np.uint8)  # red gradient
    arr[:, :, 1] = 128
    arr[:, :, 2] = 64
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def rgba_image(rgb_image: Image.Image) -> Image.Image:
    """A small RGBA test image (RGB + full-opaque alpha)."""
    return rgb_image.convert("RGBA")


@pytest.fixture
def alpha_image() -> Image.Image:
    """A 100x80 grayscale alpha matte with a centered white circle."""
    arr = np.zeros((80, 100), dtype=np.uint8)
    cy, cx = 40, 50
    for y in range(80):
        for x in range(100):
            if (x - cx) ** 2 + (y - cy) ** 2 < 30**2:
                arr[y, x] = 255
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def uniform_alpha() -> Image.Image:
    """A 100x80 uniform mid-gray alpha matte (value=128)."""
    return Image.new("L", (100, 80), 128)