"""Catalog-decoupling tests for genblaze-runway 0.3.0.

Coverage:
- ``DiscoverySupport.NONE`` declared on RunwayProvider.
- Family-pattern resolution covers current + future Gen variants.
- ``validate_model`` returns OK_PROVISIONAL for matched slugs (no
  probe, no discovery — honest signal).
- Pricing-removed contract (compute_cost returns None unless
  user-registered).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from genblaze_core.providers import (
    DiscoverySupport,
    ValidationOutcome,
    ValidationSource,
)


@pytest.fixture(autouse=True)
def _patch_runwayml():
    """Avoid importing the real runwayml package in tests."""
    with patch.dict("sys.modules", {"runwayml": MagicMock()}):
        yield


# --- DiscoverySupport declaration ------------------------------------------


class TestDiscoverySupportDeclaration:
    def test_runway_declares_none(self) -> None:
        """Runway has no /v1/models endpoint; the small fixed catalog
        plus SDK error surfacing is enough — no probe traffic."""
        from genblaze_runway import RunwayProvider

        assert RunwayProvider.discovery_support is DiscoverySupport.NONE


# --- Family resolution -----------------------------------------------------


class TestRunwayGenFamily:
    def test_current_models_match(self) -> None:
        """Both shipped Runway slugs route to the gen-video family."""
        from genblaze_runway import RunwayProvider

        provider = RunwayProvider(api_secret="test")
        for slug in ("gen4_turbo", "gen3a_turbo"):
            match = provider._models.match_family(slug)
            assert match is not None, slug
            assert match.family.name == "runway-gen-video", slug

    def test_future_variants_inherit(self) -> None:
        """The pattern absorbs future variants (gen5_turbo, gen4a_turbo,
        gen10_turbo, etc.) without an SDK release. This is the core
        promise of catalog decoupling."""
        from genblaze_runway import RunwayProvider

        provider = RunwayProvider(api_secret="test")
        for slug in ("gen5_turbo", "gen4a_turbo", "gen10_turbo"):
            match = provider._models.match_family(slug)
            assert match is not None, slug
            assert match.family.name == "runway-gen-video", slug

    def test_unrelated_slug_no_match(self) -> None:
        from genblaze_runway import RunwayProvider

        provider = RunwayProvider(api_secret="test")
        # Note: ``gen4_pro`` doesn't end in ``_turbo`` so doesn't match.
        for slug in ("gen4_pro", "stable-diffusion-xl", "dalle-3", "kling-image2video"):
            match = provider._models.match_family(slug)
            assert match is None, slug

    def test_resolved_spec_carries_constraints(self) -> None:
        """The family's constraints (duration ∈ {5, 10}, ratio ∈ {16:9,
        9:16}) ride on every resolved spec. Subclasses inherit Runway's
        validation without code duplication."""
        from genblaze_runway import RunwayProvider

        provider = RunwayProvider(api_secret="test")
        spec = provider._models.get("gen4_turbo")
        assert len(spec.param_constraints) == 2
        # The aspect_ratio → ratio alias travels with the spec_template.
        assert spec.param_aliases.get("aspect_ratio") == "ratio"


# --- validate_model end-to-end --------------------------------------------


class TestValidateModel:
    def test_family_matched_slug_provisional(self) -> None:
        """``DiscoverySupport.NONE`` + family match → OK_PROVISIONAL.
        Pipeline preflight emits a one-time WARN; doesn't raise."""
        from genblaze_runway import RunwayProvider

        provider = RunwayProvider(api_secret="test")
        result = provider.validate_model("gen4_turbo")
        assert result.outcome is ValidationOutcome.OK_PROVISIONAL
        assert result.family_name == "runway-gen-video"

    def test_unmatched_slug_unknown_permissive(self) -> None:
        """No family match + DiscoverySupport.NONE → UNKNOWN_PERMISSIVE.
        Pipeline preflight emits a one-time WARN."""
        from genblaze_runway import RunwayProvider

        provider = RunwayProvider(api_secret="test")
        result = provider.validate_model("not-a-runway-slug")
        assert result.outcome is ValidationOutcome.UNKNOWN_PERMISSIVE

    def test_user_registered_authoritative(self) -> None:
        """A user-registered exact spec shortcuts the family path and
        returns OK_AUTHORITATIVE — the SDK has positive confirmation."""
        from genblaze_core.models.enums import Modality
        from genblaze_core.providers import ModelSpec
        from genblaze_runway import RunwayProvider

        provider = RunwayProvider(api_secret="test")
        # Fork before mutating; otherwise the registration leaks across
        # tests via the class-level models_default() cache.
        provider._models = provider.models.fork()
        provider.models.register(ModelSpec(model_id="custom-runway", modality=Modality.VIDEO))
        result = provider.validate_model("custom-runway")
        assert result.outcome is ValidationOutcome.OK_AUTHORITATIVE
        assert result.source is ValidationSource.USER


# --- Pricing-removed contract --------------------------------------------


class TestPricingPhaseOut:
    def test_default_spec_has_no_pricing(self) -> None:
        """The family's spec_template carries no pricing — the SDK is
        out of the rate-curation business."""
        from genblaze_runway import RunwayProvider

        provider = RunwayProvider(api_secret="test")
        spec = provider._models.get("gen4_turbo")
        assert spec.pricing is None
