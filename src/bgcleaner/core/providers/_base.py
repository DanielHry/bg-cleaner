"""Base types and shared utilities for matting providers.

This module defines the ``ModelCard`` schema (what a model *is*), the
``MattingModel`` protocol (what a model *does*), and low-level ONNX
helpers reused across all provider implementations.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import onnxruntime as ort
from PIL import Image
from pydantic import BaseModel, Field

from bgcleaner.errors import ModelInferenceError, ModelLoadError
from bgcleaner.schemas import MattingInput, MattingOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Windows DLL discovery for CUDA / cuDNN pip packages
# ---------------------------------------------------------------------------

_dll_dirs_registered = False


def _register_nvidia_dll_dirs() -> None:
    """Register NVIDIA pip-installed DLL directories on Windows.

    The ``nvidia-cudnn-cu12`` and ``nvidia-cublas-cu12`` pip packages
    install their DLLs under ``site-packages/nvidia/<lib>/bin/``.
    ONNX Runtime loads its CUDA/cuDNN dependencies via the native
    Windows DLL loader, which only searches ``PATH`` — so
    ``os.add_dll_directory()`` alone is not sufficient.

    This function prepends the relevant directories to ``PATH`` and
    also calls ``os.add_dll_directory()`` for completeness.  It is
    called once, lazily, before the first ONNX session is created.
    It is a no-op on Linux/macOS or if the packages are not installed.
    """
    global _dll_dirs_registered  # noqa: PLW0603
    if _dll_dirs_registered or sys.platform != "win32":
        _dll_dirs_registered = True
        return

    _dll_dirs_registered = True

    nvidia_libs = ("cudnn", "cublas", "cuda_runtime", "cufft", "curand")
    added: list[str] = []
    for lib in nvidia_libs:
        try:
            mod = __import__(f"nvidia.{lib}", fromlist=[lib])
            bin_dir = Path(mod.__path__[0]) / "bin"
            if bin_dir.is_dir():
                bin_str = str(bin_dir)
                # Prepend to PATH so the native loader finds the DLLs.
                os.environ["PATH"] = bin_str + os.pathsep + os.environ.get("PATH", "")
                # Also register for Python's own DLL search.
                os.add_dll_directory(bin_str)
                added.append(bin_str)
        except (ImportError, AttributeError, IndexError):
            continue

    if added:
        logger.info("Registered %d NVIDIA DLL directories in PATH", len(added))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ModelCard(BaseModel):
    """Metadata describing a single model variant.

    Each provider module exposes a ``MODEL_CARDS`` list. The registry
    uses these cards to match ONNX files on disk to engine classes.

    Attributes:
        id: Unique machine-readable identifier (e.g. ``"rmbg2_int8"``).
        name: Human-readable display name for the UI.
        filename: Expected ONNX filename inside the models directory.
        description: Short description shown in the UI tooltip.
    """

    model_config = {"frozen": True}

    id: str
    name: str
    filename: str
    description: str = ""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MattingModel(Protocol):
    """Interface that every matting engine must implement."""

    def predict(self, input_data: MattingInput) -> MattingOutput:
        """Run matting on a single image.

        Args:
            input_data: Validated input containing a PIL image.

        Returns:
            A ``MattingOutput`` with the alpha matte resized to the
            original image dimensions.
        """
        ...


# ---------------------------------------------------------------------------
# Shared ONNX helpers
# ---------------------------------------------------------------------------


def load_onnx_session(
    model_path: Path,
    providers: list[str],
    label: str = "model",
) -> ort.InferenceSession:
    """Load an ONNX model into an inference session.

    Args:
        model_path: Filesystem path to the ``.onnx`` weights.
        providers: Ordered list of ONNX execution providers.
        label: Human-readable model name for log messages.

    Returns:
        A ready-to-use ``InferenceSession``.

    Raises:
        ModelLoadError: On any loading failure.
    """
    if not model_path.is_file():
        raise ModelLoadError(f"Model file not found: {model_path}")
    _register_nvidia_dll_dirs()
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load ONNX model from {model_path}: {exc}"
        ) from exc

    logger.info("%s loaded from %s", label, model_path)
    return session


def run_onnx(
    session: ort.InferenceSession,
    input_name: str,
    tensor: np.ndarray,
) -> np.ndarray:
    """Execute an ONNX session and return the first output.

    Args:
        session: A loaded ``InferenceSession``.
        input_name: Name of the model's input tensor.
        tensor: Preprocessed input array.

    Returns:
        The first output tensor from the model.

    Raises:
        ModelInferenceError: On any runtime failure.
    """
    try:
        outputs = session.run(None, {input_name: tensor})
    except Exception as exc:
        raise ModelInferenceError(f"ONNX inference failed: {exc}") from exc
    return outputs[0]


def raw_alpha_to_pil(
    raw_alpha: np.ndarray,
    original_size: tuple[int, int],
    apply_sigmoid: bool = False,
) -> Image.Image:
    """Convert raw model output to a PIL alpha matte.

    Args:
        raw_alpha: Model output array (any shape with H×W as last two
            dimensions — leading batch/channel dims are squeezed).
        original_size: Target ``(width, height)`` for resizing.
        apply_sigmoid: If ``True``, apply sigmoid before scaling.
            Required for models that output raw logits.

    Returns:
        A PIL image in mode ``L`` matching *original_size*.
    """
    alpha_2d = raw_alpha.squeeze()
    if alpha_2d.ndim != 2:
        alpha_2d = alpha_2d[0]

    if apply_sigmoid:
        alpha_2d = 1.0 / (1.0 + np.exp(-alpha_2d.astype(np.float32)))

    # Stretch to full [0, 1] range — some models (e.g. RMBG-2.0)
    # output a compressed range even after sigmoid.
    a_min, a_max = alpha_2d.min(), alpha_2d.max()
    if a_max - a_min > 1e-6:
        alpha_2d = (alpha_2d - a_min) / (a_max - a_min)

    alpha_2d = np.clip(alpha_2d, 0.0, 1.0)
    alpha_uint8 = (alpha_2d * 255).astype(np.uint8)

    alpha_image = Image.fromarray(alpha_uint8, mode="L")
    alpha_image = alpha_image.resize(original_size, Image.BILINEAR)
    return alpha_image


def round_to(value: int, multiple: int) -> int:
    """Round *value* to the nearest multiple of *multiple*.

    Args:
        value: The integer to round.
        multiple: The rounding base (must be > 0).

    Returns:
        The rounded integer (at least *multiple*).
    """
    return max(multiple, (value + multiple // 2) // multiple * multiple)