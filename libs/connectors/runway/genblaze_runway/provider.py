"""RunwayProvider — adapter for the Runway Gen video API.

Uses the runwayml Python SDK with async task-based workflow:
  client.image_to_video.create() → poll task → get output URL

**Catalog architecture (genblaze-core 0.3.0):** the SDK ships pattern-keyed
``ModelFamily`` rules rather than a hardcoded slug list. The Runway Gen
family captures any ``gen<N>[a]_turbo`` slug — Gen-3, Gen-3a, Gen-4,
plus future variants (Gen-5, Gen-4a, etc.) inherit the same param shape
without an SDK release.

**DiscoverySupport.NONE**: Runway has no ``GET /v1/models`` endpoint and
the runwayml SDK doesn't expose raw HTTP for the empty-payload probe
pattern. The catalog is small and stable; submit-time errors are the
authoritative liveness signal. Pipeline preflight emits
``OK_PROVISIONAL`` (matched a family) or ``UNKNOWN_PERMISSIVE`` for
unrecognized slugs — neither raises pre-flight.

**Pricing**: Runway was previously hardcoded as ``(model, duration) →
USD`` (Gen-4 Turbo: $0.50 / $1.00 for 5s / 10s; Gen-3a Turbo: $0.25 /
$0.50). As of 0.3.0 the SDK no longer ships pricing — register the
recipe yourself if you want cost tracking. See
``docs/reference/pricing-recipes.md`` for the canonical Runway recipe.

Docs: https://docs.runwayml.com/
"""

from __future__ import annotations

import re
from typing import Any

from genblaze_core.exceptions import ProviderError
from genblaze_core.models.asset import Asset, VideoMetadata
from genblaze_core.models.enums import Modality, ProviderErrorCode
from genblaze_core.models.step import Step
from genblaze_core.providers import (
    BaseProvider,
    DiscoverySupport,
    ModelFamily,
    ModelRegistry,
    ModelSpec,
    ProviderCapabilities,
    RetryPolicy,
    route_images,
    validate_asset_url,
)
from genblaze_core.providers.retry import retry_after_from_response
from genblaze_core.runnable.config import RunnableConfig

from ._errors import map_runway_error

_VALID_DURATIONS = frozenset({5, 10})
_VALID_RATIOS = frozenset({"16:9", "9:16"})


def _check_ratio(params: dict[str, Any]) -> None:
    """Validate the (post-alias) Runway-native ``ratio`` value."""
    ratio = params.get("ratio")
    if ratio is not None and ratio not in _VALID_RATIOS:
        raise ProviderError(
            f"Invalid ratio={ratio!r}. Must be one of {set(_VALID_RATIOS)}",
            error_code=ProviderErrorCode.INVALID_INPUT,
        )


def _check_duration(params: dict[str, Any]) -> None:
    """Validate ``duration`` with Runway-specific error wording."""
    if "duration" not in params:
        return
    try:
        dur = int(params["duration"])
    except (TypeError, ValueError) as exc:
        raise ProviderError(
            f"Invalid duration={params['duration']!r}. Must be one of {set(_VALID_DURATIONS)}",
            error_code=ProviderErrorCode.INVALID_INPUT,
        ) from exc
    if dur not in _VALID_DURATIONS:
        raise ProviderError(
            f"Invalid duration={dur}. Must be one of {set(_VALID_DURATIONS)}",
            error_code=ProviderErrorCode.INVALID_INPUT,
        )
    params["duration"] = dur


# Single family covering the Runway Gen video catalog. The pattern
# ``^gen\w+_turbo$`` absorbs current (gen3a_turbo, gen4_turbo) and any
# future (gen4a_turbo, gen5_turbo, etc.) variants without a code
# change. Constraints (duration ∈ {5, 10}, ratio ∈ {16:9, 9:16}) are
# Runway-wide rather than per-model, so they live on the family
# spec_template.
_RUNWAY_GEN_FAMILY = ModelFamily(
    name="runway-gen-video",
    pattern=re.compile(r"^gen\w+_turbo$"),
    spec_template=ModelSpec(
        model_id="*",
        modality=Modality.VIDEO,
        param_aliases={"aspect_ratio": "ratio"},
        param_constraints=(_check_duration, _check_ratio),
        input_mapping=route_images(slots=("prompt_image",)),
    ),
    description="Runway Gen video family (Gen-3a, Gen-4, future *_turbo variants).",
    example_slugs=("gen4_turbo", "gen3a_turbo"),
)


_FALLBACK = ModelSpec(
    model_id="*",
    modality=Modality.VIDEO,
    param_aliases={"aspect_ratio": "ratio"},
    input_mapping=route_images(slots=("prompt_image",)),
)


class RunwayProvider(BaseProvider):
    """Provider adapter for Runway video generation (Gen-3, Gen-4).

    Models match the ``runway-gen-video`` family (any ``gen<N>[a]_turbo``
    slug). Duration must be 5 or 10; aspect ratio must be 16:9 or 9:16.

    Auth: Set ``RUNWAYML_API_SECRET`` env var or pass ``api_secret``.

    Args:
        api_secret: Runway API secret. Falls back to ``RUNWAYML_API_SECRET``
            env var.
        poll_interval: Seconds between task status polls (default 5).
        models: Optional custom ``ModelRegistry`` — overrides the class default.
        retry_policy: Optional retry policy override.
        probe_cache_ttl: Per-instance TTL for the probe cache (unused for
            ``DiscoverySupport.NONE`` providers but accepted for ctor
            compatibility with sibling connectors).
        probe_cache_max_entries: Per-instance probe-cache size cap.
    """

    name = "runway"
    discovery_support = DiscoverySupport.NONE
    """Runway has no ``GET /v1/models`` endpoint and the runwayml SDK
    doesn't expose raw HTTP for the empty-payload probe pattern. The
    catalog is small (2 models today) and stable — submit-time errors
    are the authoritative liveness signal. Pipeline preflight emits
    ``OK_PROVISIONAL`` for family-matched slugs."""

    @classmethod
    def create_registry(cls) -> ModelRegistry:
        return ModelRegistry(
            provider_families=(_RUNWAY_GEN_FAMILY,),
            fallback=_FALLBACK,
        )

    def get_capabilities(self) -> ProviderCapabilities:
        """Runway: video generation from text and/or image inputs."""
        return ProviderCapabilities(
            supported_modalities=[Modality.VIDEO],
            supported_inputs=["text", "image"],
            accepts_chain_input=True,
            max_duration=10.0,
            models=self._models.known(),
            output_formats=["video/mp4"],
        )

    def __init__(
        self,
        api_secret: str | None = None,
        poll_interval: float = 5.0,
        *,
        models: ModelRegistry | None = None,
        retry_policy: RetryPolicy | None = None,
        probe_cache_ttl: float | None = None,
        probe_cache_max_entries: int | None = None,
    ):
        super().__init__(
            models=models,
            retry_policy=retry_policy,
            probe_cache_ttl=probe_cache_ttl,
            probe_cache_max_entries=probe_cache_max_entries,
        )
        self.poll_interval = poll_interval
        self._api_secret = api_secret
        self._client: Any = None
        # Cache of in-progress task objects keyed by prediction_id, populated
        # in ``poll()`` and consumed in ``poll_progress()`` so we don't double
        # the API call rate just to surface preview/progress.
        self._progress_cache: dict[str, Any] = {}

    def normalize_params(self, params: dict, modality: Any = None) -> dict:
        """Map standard params to Runway-native names.

        Kept for backward compatibility with callers that invoke it directly;
        ``prepare_payload`` also performs the alias via the model spec.
        """
        p = dict(params)
        if "aspect_ratio" in p and "ratio" not in p:
            p["ratio"] = p.pop("aspect_ratio")
        return p

    def _get_client(self):
        if self._client is None:
            try:
                from runwayml import RunwayML
            except ImportError as exc:
                raise ProviderError(
                    "runwayml package not installed. Run: pip install runwayml"
                ) from exc
            kwargs: dict = {}
            if self._api_secret:
                kwargs["api_key"] = self._api_secret
            self._client = RunwayML(**kwargs)
        return self._client

    def submit(self, step: Step, config: RunnableConfig | None = None) -> Any:
        """Create a video generation task."""
        client = self._get_client()
        try:
            payload = self.prepare_payload(step)

            # Translate canonical 'prompt' to Runway's 'prompt_text'; only the
            # SDK-recognized keys are forwarded to image_to_video.create.
            request: dict = {
                "model": step.model,
                "prompt_text": payload.get("prompt", step.prompt or ""),
            }
            for key in ("duration", "ratio", "seed", "watermark", "prompt_image"):
                if key in payload:
                    request[key] = payload[key]
            if "watermark" in request:
                request["watermark"] = bool(request["watermark"])

            task = client.image_to_video.create(**request)
            return task.id
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Runway submit failed: {exc}",
                error_code=map_runway_error(exc),
                retry_after=retry_after_from_response(exc),
            ) from exc

    def poll(self, prediction_id: Any, config: RunnableConfig | None = None) -> bool:
        """Check if the Runway task is complete."""
        client = self._get_client()
        try:
            task = client.tasks.retrieve(prediction_id)
            if task.status in ("SUCCEEDED", "FAILED"):
                self._cache_poll_result(prediction_id, task)
                return True
            # Stash the in-progress task so poll_progress() can read it
            # without a second API call.
            self._progress_cache[str(prediction_id)] = task
            return False
        except Exception as exc:
            raise ProviderError(
                f"Runway poll failed: {exc}",
                error_code=map_runway_error(exc),
                retry_after=retry_after_from_response(exc),
            ) from exc

    def poll_progress(self, prediction_id: Any) -> dict[str, Any] | None:
        """Surface Runway task ``progress`` and any preview thumbnail.

        Reads the in-progress task cached by the most recent ``poll()`` so
        we don't hit the API twice per tick. Returns None when no task has
        been cached yet (first poll attempt) or when neither field is set.
        """
        task = self._progress_cache.get(str(prediction_id))
        if task is None:
            return None
        signals: dict[str, Any] = {}
        progress = getattr(task, "progress", None)
        if isinstance(progress, (int, float)) and 0 <= progress <= 1:
            signals["progress_pct"] = float(progress)
        # Runway's task object exposes ``thumbnail_url`` on some Gen-4 models
        # for in-progress draft frames; getattr is defensive against SDK
        # versions that don't carry the field.
        preview = getattr(task, "thumbnail_url", None) or getattr(task, "preview_url", None)
        if preview:
            signals["preview_url"] = str(preview)
        return signals or None

    def fetch_output(self, prediction_id: Any, step: Step) -> Step:
        """Fetch the completed video URL."""
        client = self._get_client()
        try:
            task = self._get_cached_poll_result(prediction_id)
            if task is None:
                task = client.tasks.retrieve(prediction_id)

            step.provider_payload = {
                "runway": {
                    "task_id": task.id,
                    "status": task.status,
                }
            }

            if task.status == "FAILED":
                error_msg = getattr(task, "failure", None) or "Video generation failed"
                raise ProviderError(
                    str(error_msg),
                    error_code=ProviderErrorCode.UNKNOWN,
                )

            # Task output contains the video URL
            output = getattr(task, "output", None)
            if output and isinstance(output, list) and len(output) > 0:
                url = str(output[0])
                validate_asset_url(url)
                step.assets.append(Asset(url=url, media_type="video/mp4"))
            elif output and isinstance(output, str):
                validate_asset_url(output)
                step.assets.append(Asset(url=output, media_type="video/mp4"))
            else:
                raise ProviderError("Runway task completed but no output URL found")

            # Default duration is 5s when the user didn't specify one — kept
            # so VideoMetadata.duration is populated consistently regardless
            # of whether the caller supplied the param. (Pricing was
            # previously also keyed off this; no longer SDK state.)
            duration = int(step.params.get("duration", 5))
            step.params.setdefault("duration", duration)
            for a in step.assets:
                a.video = VideoMetadata(has_audio=False)
                a.duration = a.duration or float(duration)

            self._apply_registry_pricing(step)
            return step
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Runway fetch_output failed: {exc}",
                error_code=map_runway_error(exc),
                retry_after=retry_after_from_response(exc),
            ) from exc
