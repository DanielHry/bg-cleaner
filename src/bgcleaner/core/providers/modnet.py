"""MODNet matting provider.

Lightweight portrait matting model (~25 MB). Best for fast inference on
portraits with clean backgrounds.

To add this model, place ``modnet.onnx`` in the models directory.
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
    round_to,
    run_onnx,
)
from bgcleaner.schemas import MattingInput, MattingOutput

MODEL_CARDS: list[ModelCard] = [
    ModelCard(
        id="modnet",
        name="MODNet",
        filename="modnet.onnx",
        description="Lightweight portrait matting (~25 MB, fast).",
    ),
]


class MODNetEngine:
    """ONNX Runtime wrapper for the MODNet portrait matting model.

    Preprocessing: shortest edge → ``modnet_ref_size`` (rounded to ×32),
    normalised to ``[-1, 1]``.

    Args:
        model_path: Path to the ONNX weights file.
        settings: Application settings.
    """

    def __init__(self, model_path: Path, settings: Settings) -> None:
        self._settings = settings
        self._session = load_onnx_session(
            model_path, settings.onnx_providers, "MODNet"
        )
        self._input_name: str = self._session.get_inputs()[0].name

    def predict(self, input_data: MattingInput) -> MattingOutput:
        """Run portrait matting on a single image.

        Args:
            input_data: Validated input containing a PIL image.

        Returns:
            Alpha matte resized to the original dimensions.
        """
        original_size = input_data.image.size
        tensor = self._preprocess(input_data.image)
        raw_alpha = run_onnx(self._session, self._input_name, tensor)
        alpha_image = raw_alpha_to_pil(raw_alpha, original_size)
        return MattingOutput(alpha=alpha_image, original_size=original_size)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize and normalise for MODNet.

        Args:
            image: Input PIL image (any mode).

        Returns:
            Float32 array of shape ``(1, 3, H, W)`` in ``[-1, 1]``.
        """
        img = image.convert("RGB")
        width, height = img.size

        ref = self._settings.modnet_ref_size
        if width < height:
            new_w = ref
            new_h = int(ref * height / width)
        else:
            new_h = ref
            new_w = int(ref * width / height)

        new_w = round_to(new_w, 32)
        new_h = round_to(new_h, 32)

        img = img.resize((new_w, new_h), Image.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        arr = arr.transpose(2, 0, 1)
        return arr[np.newaxis, ...]


def create(model_path: Path, settings: Settings) -> MODNetEngine:
    """Factory function called by the provider registry.

    Args:
        model_path: Path to the ONNX weights.
        settings: Application settings.

    Returns:
        A ready-to-use ``MODNetEngine``.
    """
    return MODNetEngine(model_path=model_path, settings=settings)