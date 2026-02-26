"""BRIA RMBG-2.0 matting provider.

High-quality background removal based on the BiRefNet architecture.
Multiple quantization variants share the same preprocessing and
postprocessing logic — only the weights differ.

To add a variant, place the corresponding ``.onnx`` file in the models
directory using the filename listed in ``MODEL_CARDS``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from bgcleaner.config import Settings
from bgcleaner.core.providers._base import (
    ModelCard,
    load_onnx_session,
    raw_alpha_to_pil,
    run_onnx,
)
from bgcleaner.schemas import MattingInput, MattingOutput

# ImageNet normalisation constants.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_CARDS: list[ModelCard] = [
    ModelCard(
        id="rmbg2",
        name="RMBG-2.0",
        filename="RMBG-2.0.onnx",
        description="High-quality background removal, FP32 (~1 GB).",
    ),
    ModelCard(
        id="rmbg2_int8",
        name="RMBG-2.0 (INT8)",
        filename="RMBG-2.0_int8.onnx",
        description="Quantized RMBG-2.0, faster on CPU (~366 MB).",
    ),
    ModelCard(
        id="rmbg2_fp16",
        name="RMBG-2.0 (FP16)",
        filename="RMBG-2.0_fp16.onnx",
        description="Half-precision RMBG-2.0, best for GPU (~514 MB).",
    ),
]


class RMBGEngine:
    """ONNX Runtime wrapper for BRIA RMBG-2.0 and its quantized variants.

    Preprocessing: fixed resize to ``rmbg_input_size`` (default 1024×1024),
    ImageNet normalisation. Postprocessing applies sigmoid to raw logits.

    Args:
        model_path: Path to the ONNX weights file.
        settings: Application settings.
    """

    def __init__(self, model_path: Path, settings: Settings) -> None:
        self._settings = settings
        self._session = load_onnx_session(
            model_path, settings.onnx_providers, "RMBG-2.0"
        )
        self._input_name: str = self._session.get_inputs()[0].name

    def predict(self, input_data: MattingInput) -> MattingOutput:
        """Run background removal on a single image.

        Args:
            input_data: Validated input containing a PIL image.

        Returns:
            Alpha matte resized to the original dimensions.
        """
        original_size = input_data.image.size
        tensor = self._preprocess(input_data.image)
        raw_output = run_onnx(self._session, self._input_name, tensor)
        alpha_image = raw_alpha_to_pil(
            raw_output, original_size, apply_sigmoid=True
        )
        return MattingOutput(alpha=alpha_image, original_size=original_size)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize and normalise for RMBG-2.0.

        Args:
            image: Input PIL image (any mode).

        Returns:
            Float32 array of shape ``(1, 3, 1024, 1024)`` with ImageNet
            normalisation.
        """
        img = image.convert("RGB")
        img = img.resize(self._settings.rmbg_input_size, Image.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD

        arr = arr.transpose(2, 0, 1)
        return arr[np.newaxis, ...]


def create(model_path: Path, settings: Settings) -> RMBGEngine:
    """Factory function called by the provider registry.

    Args:
        model_path: Path to the ONNX weights.
        settings: Application settings.

    Returns:
        A ready-to-use ``RMBGEngine``.
    """
    return RMBGEngine(model_path=model_path, settings=settings)