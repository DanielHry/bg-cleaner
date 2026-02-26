"""Provider registry — discovers models on disk and creates engines.

Adding a new model requires two steps:

1. Create a provider module in this package (e.g. ``my_model.py``) that
   exports ``MODEL_CARDS`` (list of ``ModelCard``) and a ``create``
   factory function with signature
   ``(model_path: Path, settings: Settings) -> MattingModel``.
2. Import the module in ``_PROVIDERS`` below.

The rest (UI listing, engine creation) is handled automatically.

Typical usage::

    from bgcleaner.core.providers import discover_available, create_engine

    cards = discover_available(settings)
    engine = create_engine(cards[0], settings)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from bgcleaner.config import Settings
from bgcleaner.core.providers._base import MattingModel, ModelCard
from bgcleaner.core.providers import modnet, rmbg

logger = logging.getLogger(__name__)

# Type alias for a provider factory function.
ProviderFactory = Callable[[Path, Settings], MattingModel]

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# Maps model card id → (ModelCard, factory function).
# To add a new provider, import its module and add its entries here.
_REGISTRY: dict[str, tuple[ModelCard, ProviderFactory]] = {}


def _register_provider_module(
    cards: list[ModelCard],
    factory: ProviderFactory,
) -> None:
    """Register all model cards from a provider module.

    Args:
        cards: The ``MODEL_CARDS`` list exported by the provider.
        factory: The ``create`` function exported by the provider.
    """
    for card in cards:
        if card.id in _REGISTRY:
            logger.warning("Duplicate model id '%s' — skipping.", card.id)
            continue
        _REGISTRY[card.id] = (card, factory)


# Register built-in providers.
_register_provider_module(modnet.MODEL_CARDS, modnet.create)
_register_provider_module(rmbg.MODEL_CARDS, rmbg.create)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_all_cards() -> list[ModelCard]:
    """Return every registered model card (whether the file exists or not).

    Returns:
        A list of all known ``ModelCard`` instances.
    """
    return [card for card, _ in _REGISTRY.values()]


def discover_available(settings: Settings) -> list[ModelCard]:
    """Return model cards whose ONNX file is present in ``models_dir``.

    Args:
        settings: Application settings (provides ``models_dir``).

    Returns:
        A list of ``ModelCard`` instances for which the corresponding
        ``.onnx`` file was found on disk, sorted by display name.
    """
    models_dir = settings.models_dir
    available: list[ModelCard] = []

    for card, _ in _REGISTRY.values():
        if (models_dir / card.filename).is_file():
            available.append(card)

    available.sort(key=lambda c: c.name)
    return available


def create_engine(card: ModelCard, settings: Settings) -> MattingModel:
    """Instantiate the engine for a specific model card.

    Args:
        card: The model card to load (must be registered).
        settings: Application settings.

    Returns:
        A ``MattingModel`` implementation ready for inference.

    Raises:
        ValueError: If the card id is not registered.
        ModelLoadError: If the ONNX file cannot be loaded.
    """
    if card.id not in _REGISTRY:
        raise ValueError(
            f"Unknown model id '{card.id}'. "
            f"Registered: {sorted(_REGISTRY.keys())}"
        )

    _, factory = _REGISTRY[card.id]
    model_path = settings.models_dir / card.filename
    return factory(model_path, settings)


# Re-export key types for convenience.
__all__ = [
    "MattingModel",
    "ModelCard",
    "create_engine",
    "discover_available",
    "get_all_cards",
]