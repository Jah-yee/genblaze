"""Google model families — Veo (video) and Imagen (image).

Pattern-keyed; future ``veo-N`` / ``imagen-N`` slugs inherit param
shape automatically. Both families ship the
``google_models_get_probe`` so ``Pipeline.preflight()`` returns
``OK_AUTHORITATIVE`` when the slug exists on the user's project,
``NOT_FOUND`` when it doesn't, and ``OK_PROVISIONAL`` when google-genai
returns a non-404 error (auth, region, transport).

Pricing intentionally absent: see
``docs/reference/pricing-recipes.md`` for Veo (per-second by model)
and Imagen (per-image by model) recipes.
"""

from __future__ import annotations

import re
from typing import Any

from genblaze_core.exceptions import ProviderError
from genblaze_core.models.enums import Modality, ProviderErrorCode
from genblaze_core.providers import (
    ModelFamily,
    ModelSpec,
)

from ._probe import google_models_get_probe

# --- Constraint helpers (constructor-cheap, family-shared) ----------------

_VEO_VALID_ASPECT_RATIOS = frozenset({"16:9", "9:16"})
_VEO_VALID_RESOLUTIONS = frozenset({"720p", "1080p", "4k"})
_VEO_VALID_DURATIONS = frozenset({"4", "6", "8"})
_IMAGEN_VALID_ASPECT_RATIOS = frozenset({"1:1", "3:4", "4:3", "9:16", "16:9"})


def _check_veo_aspect_ratio(params: dict[str, Any]) -> None:
    ar = params.get("aspect_ratio")
    if ar is not None and ar not in _VEO_VALID_ASPECT_RATIOS:
        raise ProviderError(
            f"Invalid aspect_ratio={ar!r}. Must be one of {set(_VEO_VALID_ASPECT_RATIOS)}",
            error_code=ProviderErrorCode.INVALID_INPUT,
        )


def _check_veo_resolution(params: dict[str, Any]) -> None:
    res = params.get("resolution")
    if res is not None and res not in _VEO_VALID_RESOLUTIONS:
        raise ProviderError(
            f"Invalid resolution={res!r}. Must be one of {set(_VEO_VALID_RESOLUTIONS)}",
            error_code=ProviderErrorCode.INVALID_INPUT,
        )


def _check_veo_duration(params: dict[str, Any]) -> None:
    if "duration_seconds" not in params:
        return
    dur = params["duration_seconds"]
    if dur not in _VEO_VALID_DURATIONS:
        raise ProviderError(
            f"Invalid duration_seconds={dur!r}. Must be one of {set(_VEO_VALID_DURATIONS)}",
            error_code=ProviderErrorCode.INVALID_INPUT,
        )


def _check_imagen_aspect_ratio(params: dict[str, Any]) -> None:
    ar = params.get("aspect_ratio")
    if ar is not None and ar not in _IMAGEN_VALID_ASPECT_RATIOS:
        raise ProviderError(
            f"Invalid aspect_ratio={ar!r}. Must be one of {_IMAGEN_VALID_ASPECT_RATIOS}",
            error_code=ProviderErrorCode.INVALID_INPUT,
        )


# --- Families -------------------------------------------------------------

GOOGLE_VEO_FAMILY = ModelFamily(
    name="google-veo",
    pattern=re.compile(r"^veo-"),
    spec_template=ModelSpec(
        model_id="*",
        modality=Modality.VIDEO,
        # Standard "duration" maps to Veo-native "duration_seconds".
        param_aliases={"duration": "duration_seconds"},
        # Veo expects "4"/"6"/"8" string form.
        param_coercers={"duration_seconds": str},
        param_constraints=(
            _check_veo_aspect_ratio,
            _check_veo_resolution,
            _check_veo_duration,
        ),
    ),
    description=(
        "Google Veo family — text/image-to-video generation. Veo 3 variants "
        "additionally render synchronized audio."
    ),
    example_slugs=(
        "veo-2.0-generate-001",
        "veo-3.0-generate-001",
        "veo-3.0-fast-generate-001",
    ),
    probe=google_models_get_probe,
)


GOOGLE_IMAGEN_FAMILY = ModelFamily(
    name="google-imagen",
    pattern=re.compile(r"^imagen-"),
    spec_template=ModelSpec(
        model_id="*",
        modality=Modality.IMAGE,
        param_constraints=(_check_imagen_aspect_ratio,),
    ),
    description="Google Imagen family — text-to-image generation.",
    example_slugs=(
        "imagen-3.0-generate-002",
        "imagen-3.0-fast-generate-001",
    ),
    probe=google_models_get_probe,
)


__all__ = ["GOOGLE_VEO_FAMILY", "GOOGLE_IMAGEN_FAMILY"]
