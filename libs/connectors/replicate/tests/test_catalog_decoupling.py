"""Catalog-decoupling regression tests for ``genblaze-replicate``.

Replicate is the proof-point for ``DiscoverySupport.NATIVE`` with an
empty default registry: every slug routes through the permissive
fallback spec, and the upstream catalog is enumerated dynamically via
``client.models.list()`` (wired in ``__init__`` per-instance for
auth-aware caches).

This file pins the contract for that minimal surface so a future
edit can't silently regress the post-decoupling shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from genblaze_core.providers import DiscoverySupport


@pytest.fixture(autouse=True)
def _patch_replicate_sdk():
    """Avoid importing the real replicate SDK in tests."""
    fake = MagicMock()
    with patch.dict("sys.modules", {"replicate": fake}):
        yield


# --- DiscoverySupport declaration -----------------------------------------


class TestDiscoverySupportDeclaration:
    def test_native(self) -> None:
        from genblaze_replicate import ReplicateProvider

        assert ReplicateProvider.discovery_support is DiscoverySupport.NATIVE


# --- Registry shape (no families, fallback only) --------------------------


class TestRegistryShape:
    def test_no_provider_families_shipped(self) -> None:
        """Replicate is meta-vendor — every model on its hub is a valid
        slug. The connector ships zero families and lets the upstream
        catalog speak for itself via discovery."""
        from genblaze_replicate import ReplicateProvider

        provider = ReplicateProvider(api_token="test")
        assert provider._models.families == ()

    def test_arbitrary_slug_resolves_via_fallback(self) -> None:
        """Any slug — known model, brand-new release, typo — resolves
        through the fallback. Replicate's own API gates submit-time
        liveness."""
        from genblaze_replicate import ReplicateProvider

        provider = ReplicateProvider(api_token="test")
        for slug in (
            "black-forest-labs/flux-schnell",
            "stability-ai/sdxl",
            "owner/brand-new-model",
            "a-typo-slug",
        ):
            spec = provider._models.get(slug)
            assert spec is not None, slug


# --- Pricing-removed contract --------------------------------------------


class TestPricingPhaseOut:
    def test_fallback_spec_carries_no_pricing(self) -> None:
        """The connector's fallback spec ships with ``pricing=None``;
        cost is user-registered via ``register_pricing``."""
        from genblaze_replicate import ReplicateProvider

        provider = ReplicateProvider(api_token="test")
        spec = provider._models.get("any/slug")
        assert spec.pricing is None


# --- Cross-provider isolation --------------------------------------------


class TestCrossProviderIsolation:
    def test_replicate_does_not_match_other_provider_slugs(self) -> None:
        """Replicate has no families — ``match_family`` returns None
        for any slug; nothing matches, including slugs that look like
        another provider's catalog."""
        from genblaze_replicate import ReplicateProvider

        provider = ReplicateProvider(api_token="test")
        for slug in (
            "veo-3.0-generate-001",
            "imagen-3.0-generate-002",
            "tts-1",
            "ray-2",
        ):
            assert provider._models.match_family(slug) is None, slug
