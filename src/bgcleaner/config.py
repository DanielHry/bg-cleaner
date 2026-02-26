"""Application settings loaded from environment and .env files."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Global configuration for the BG-cleaner application.

    Values are loaded in order: field defaults → .env file → environment
    variables. Environment variables are prefixed with ``BGC_``.

    Attributes:
        models_dir: Directory containing ONNX model files. The provider
            registry scans this directory to discover available models.
        onnx_providers: ONNX Runtime execution providers, tried in order.
        modnet_ref_size: Reference size for MODNet preprocessing (shortest
            edge, rounded to nearest multiple of 32).
        rmbg_input_size: Fixed input resolution for RMBG-2.0.
        supported_formats: Allowed upload image extensions (lowercase,
            without dot).
        max_upload_mb: Maximum upload file size in megabytes.
        alpha_brightness_range: Min/max bounds for the brightness slider.
        alpha_contrast_range: Min/max bounds for the contrast slider.
        alpha_sharpness_range: Min/max bounds for the sharpness slider.
        alpha_blur_max: Maximum Gaussian blur radius in pixels.
    """

    model_config = SettingsConfigDict(
        env_prefix="BGC_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- Models ---
    models_dir: Path = _PROJECT_ROOT / "assets" / "models"
    onnx_providers: list[str] = ["CPUExecutionProvider"]

    # --- Provider-specific ---
    modnet_ref_size: int = 512
    rmbg_input_size: tuple[int, int] = (1024, 1024)

    # --- Upload ---
    supported_formats: list[str] = ["jpg", "jpeg", "png", "webp"]
    max_upload_mb: float = 10.0

    # --- Alpha adjustment bounds (for UI sliders) ---
    alpha_brightness_range: tuple[float, float] = (0.0, 3.0)
    alpha_contrast_range: tuple[float, float] = (0.0, 3.0)
    alpha_sharpness_range: tuple[float, float] = (0.0, 3.0)
    alpha_blur_max: float = 10.0