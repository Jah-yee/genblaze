"""VeoProvider — adapter for Google Veo video generation.

Uses the google-genai SDK with the async operation-based workflow:
  client.models.generate_videos() → poll operation → download video

**Catalog architecture (genblaze-core 0.3.0):** the SDK ships the
pattern-keyed ``google-veo`` family (``^veo-``) instead of a
hardcoded slug list. New ``veo-N`` slugs inherit the param shape;
authoritative liveness comes from ``client.models.get(model=slug)``
via the family probe.

**Pricing**: per-second-by-model rates were dropped in 0.3.0. See
``docs/reference/pricing-recipes.md`` for the canonical Veo recipe.

Docs: https://ai.google.dev/gemini-api/docs/video
"""

from __future__ import annotations

from typing import Any

from genblaze_core.exceptions import ProviderError
from genblaze_core.models.asset import Asset, AudioMetadata, Track, VideoMetadata
from genblaze_core.models.enums import Modality, ProviderErrorCode
from genblaze_core.models.step import Step
from genblaze_core.providers import (
    BaseProvider,
    DiscoverySupport,
    LiveProbeResult,
    ModelRegistry,
    ModelSpec,
    ProviderCapabilities,
    RetryPolicy,
    validate_asset_url,
)
from genblaze_core.providers.retry import retry_after_from_response
from genblaze_core.runnable.config import RunnableConfig

from genblaze_google._errors import map_google_error
from genblaze_google._families import GOOGLE_VEO_FAMILY, GOOGLE_VEO_LEGACY_FAMILY

_FALLBACK = ModelSpec(model_id="*", modality=Modality.VIDEO)


class VeoProvider(BaseProvider):
    """Provider adapter for Google Veo video generation.

    Models match the ``google-veo`` family (``^veo-``). Current GA
    examples: ``veo-2.0-generate-001``, ``veo-3.0-generate-001``,
    ``veo-3.0-fast-generate-001``.

    Supports both Gemini API (``api_key``) and Vertex AI
    (``project``/``location``) auth.

    Args:
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        project: GCP project ID for Vertex AI auth (mutually exclusive with api_key).
        location: GCP region for Vertex AI (default "us-central1").
        poll_interval: Seconds between operation polls (default 10).
        models: Optional custom ``ModelRegistry`` — overrides the class default.
        retry_policy: Optional retry policy override.
        probe_cache_ttl: Per-instance probe-cache TTL.
        probe_cache_max_entries: Per-instance probe-cache size cap.
    """

    name = "google-veo"
    discovery_support = DiscoverySupport.PARTIAL
    """google-genai has no per-modality catalog endpoint that filters
    Veo cleanly. The family probe (``client.models.get``) is the
    authoritative liveness check; preflight surfaces dead slugs as
    ``NOT_FOUND`` before the operation submission."""

    @classmethod
    def create_registry(cls) -> ModelRegistry:
        # Order is load-bearing: legacy first, modern catch-all second.
        # ``ModelRegistry.match_family`` is first-match-wins, so a
        # ``veo-2.0-*`` slug must match ``GOOGLE_VEO_LEGACY_FAMILY``
        # (no audio) before falling through to ``GOOGLE_VEO_FAMILY``
        # (which carries ``extras["has_audio"]=True``).
        return ModelRegistry(
            provider_families=(GOOGLE_VEO_LEGACY_FAMILY, GOOGLE_VEO_FAMILY),
            fallback=_FALLBACK,
        )

    def get_capabilities(self) -> ProviderCapabilities:
        """Veo: video generation from text prompts with configurable resolution and duration."""
        return ProviderCapabilities(
            supported_modalities=[Modality.VIDEO],
            supported_inputs=["text"],
            max_duration=8.0,
            resolutions=["720p", "1080p", "4k"],
            models=self._models.known(),
            output_formats=["video/mp4"],
        )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        project: str | None = None,
        location: str = "us-central1",
        poll_interval: float = 10.0,
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
        self._api_key = api_key
        self._project = project
        self._location = location
        self._client: Any = None

    def _invoke_family_probe(self, probe: Any, model_id: str) -> LiveProbeResult:
        """Forward the family probe with this provider's lazy genai client."""
        return probe(model_id, client=self._get_client())

    def normalize_params(self, params: dict, modality: Any = None) -> dict:
        """Map standard params to Veo-native names.

        Kept for backward compatibility with direct callers; ``prepare_payload``
        also performs the alias via the model spec.
        """
        p = dict(params)
        if "duration" in p and "duration_seconds" not in p:
            p["duration_seconds"] = p.pop("duration")
        return p

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ProviderError(
                    "google-genai package not installed. Run: pip install google-genai"
                ) from exc

            if self._project:
                # Vertex AI auth
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project,
                    location=self._location,
                )
            else:
                # Gemini API key auth
                kwargs: dict = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                self._client = genai.Client(**kwargs)
        return self._client

    def _build_config(self, payload: dict[str, Any], step: Step) -> Any:
        """Build a GenerateVideosConfig from the prepared payload."""
        from google.genai import types

        config_kwargs: dict = {}

        if "aspect_ratio" in payload:
            config_kwargs["aspect_ratio"] = payload["aspect_ratio"]
        if "resolution" in payload:
            config_kwargs["resolution"] = payload["resolution"]
        if "duration_seconds" in payload:
            config_kwargs["duration_seconds"] = payload["duration_seconds"]
        if "person_generation" in payload:
            config_kwargs["person_generation"] = payload["person_generation"]
        if "number_of_videos" in payload:
            config_kwargs["number_of_videos"] = int(payload["number_of_videos"])
        if "enhance_prompt" in payload:
            config_kwargs["enhance_prompt"] = bool(payload["enhance_prompt"])
        if step.seed is not None:
            config_kwargs["seed"] = step.seed

        return types.GenerateVideosConfig(**config_kwargs) if config_kwargs else None

    def submit(self, step: Step, config: RunnableConfig | None = None) -> Any:
        """Start a video generation operation."""
        client = self._get_client()
        try:
            payload = self.prepare_payload(step)
            gen_config = self._build_config(payload, step)
            kwargs: dict = {
                "model": step.model,
                "prompt": payload.get("prompt", step.prompt or ""),
            }
            if gen_config is not None:
                kwargs["config"] = gen_config

            operation = client.models.generate_videos(**kwargs)
            # Return the provider-native operation name for resume() compatibility
            return operation.name
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Veo submit failed: {exc}",
                error_code=map_google_error(exc),
                retry_after=retry_after_from_response(exc),
            ) from exc

    def poll(self, prediction_id: Any, config: RunnableConfig | None = None) -> bool:
        """Check if the video generation operation is done."""
        client = self._get_client()
        try:
            operation = client.operations.get(prediction_id)
            if operation.done:
                self._cache_poll_result(prediction_id, operation)
                return True
            return False
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Veo poll failed: {exc}",
                error_code=map_google_error(exc),
                retry_after=retry_after_from_response(exc),
            ) from exc

    def fetch_output(self, prediction_id: Any, step: Step) -> Step:
        """Download generated video(s) and attach asset URLs."""
        client = self._get_client()
        try:
            # Use cached poll result if available, otherwise fetch fresh
            operation = self._get_cached_poll_result(prediction_id)
            if operation is None:
                operation = client.operations.get(prediction_id)

            # Store provider metadata
            step.provider_payload = {
                "google": {
                    "operation_name": getattr(operation, "name", None),
                    "model": step.model,
                }
            }

            # Check for errors in the operation result
            if hasattr(operation, "error") and operation.error:
                raise ProviderError(
                    str(operation.error),
                    error_code=ProviderErrorCode.UNKNOWN,
                )

            response = operation.response
            if response is None or not hasattr(response, "generated_videos"):
                raise ProviderError("No video generated in response")

            # Audio capability comes from the family's typed ``extras``,
            # not a runtime string check on the slug. Veo 2 routes to
            # ``GOOGLE_VEO_LEGACY_FAMILY`` (no ``has_audio``); Veo 3+
            # routes to ``GOOGLE_VEO_FAMILY`` (``extras["has_audio"]=True``).
            # Future ``veo-N`` slugs inherit modern's audio capability
            # automatically — no provider release required.
            spec = self._models.get(step.model)
            has_audio = bool(spec.extras.get("has_audio"))

            for gv in response.generated_videos:
                video = gv.video
                # Download to get the file URI
                client.files.download(file=video)
                # Use the video's URI as the asset URL
                video_uri = getattr(video, "uri", None)
                if video_uri:
                    validate_asset_url(video_uri)
                    vm_kwargs: dict[str, Any] = {"has_audio": has_audio}
                    if "resolution" in step.params:
                        vm_kwargs["resolution"] = step.params["resolution"]
                    asset = Asset(url=video_uri, media_type="video/mp4")
                    asset.video = VideoMetadata(**vm_kwargs)
                    # Multi-track metadata for audio-capable variants
                    # (video + generated audio)
                    if has_audio:
                        asset.tracks = [
                            Track(kind="video", codec="h264"),
                            Track(kind="audio", codec="aac", label="generated-audio"),
                        ]
                        asset.audio = AudioMetadata(codec="aac")
                    step.assets.append(asset)
                else:
                    # Fallback: save locally and use file path
                    raise ProviderError(
                        "Veo response missing video URI — "
                        "use client.files.download() to save locally"
                    )

            self._apply_registry_pricing(step)
            return step
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Veo fetch_output failed: {exc}",
                error_code=map_google_error(exc),
                retry_after=retry_after_from_response(exc),
            ) from exc
