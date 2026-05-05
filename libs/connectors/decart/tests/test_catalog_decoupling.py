"""Catalog-decoupling tests for genblaze-decart 0.3.0.

Coverage:
- ``DiscoverySupport.NONE`` declared on both DecartVideoProvider and
  DecartImageProvider.
- Family-pattern resolution covers current + future Lucy variants.
- Cross-modality isolation: video slugs don't match image family and
  vice versa.
- ``validate_model`` returns OK_PROVISIONAL for matched slugs.
- Pricing-removed contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from genblaze_core.providers import (
    DiscoverySupport,
    ValidationOutcome,
)


@pytest.fixture(autouse=True)
def _patch_decart():
    """Avoid importing the real decart package in tests."""
    mock_decart = MagicMock()
    with patch.dict("sys.modules", {"decart": mock_decart}):
        yield


# --- DiscoverySupport declarations ----------------------------------------


class TestDiscoverySupportDeclarations:
    def test_video_declares_none(self) -> None:
        from genblaze_decart import DecartVideoProvider

        assert DecartVideoProvider.discovery_support is DiscoverySupport.NONE

    def test_image_declares_none(self) -> None:
        from genblaze_decart import DecartImageProvider

        assert DecartImageProvider.discovery_support is DiscoverySupport.NONE


# --- Family resolution -----------------------------------------------------


class TestLucyVideoFamily:
    def test_current_video_models_match(self) -> None:
        from genblaze_decart import DecartVideoProvider

        provider = DecartVideoProvider(api_key="test")
        for slug in (
            "lucy-pro-t2v",
            "lucy-pro-i2v",
            "lucy-pro-v2v",
            "lucy-2-v2v",
            "lucy-fast-v2v",
            "lucy-motion",
            "lucy-dev-i2v",
            "lucy-restyle-v2v",
        ):
            match = provider._models.match_family(slug)
            assert match is not None, slug
            assert match.family.name == "decart-lucy-video", slug

    def test_future_lucy_video_variants_inherit(self) -> None:
        """Pattern absorbs future Lucy video slugs without code changes."""
        from genblaze_decart import DecartVideoProvider

        provider = DecartVideoProvider(api_key="test")
        for slug in ("lucy-3-v2v", "lucy-edit-v2v", "lucy-cinema-motion"):
            match = provider._models.match_family(slug)
            assert match is not None, slug
            assert match.family.name == "decart-lucy-video", slug

    def test_resolved_spec_carries_resolution_constraint(self) -> None:
        """resolution enum + enhance_prompt bool ride on the family
        spec_template — every Lucy video slug inherits the same shape."""
        from genblaze_decart import DecartVideoProvider

        provider = DecartVideoProvider(api_key="test")
        spec = provider._models.get("lucy-pro-t2v")
        assert "resolution" in spec.param_schemas
        assert "enhance_prompt" in spec.param_coercers


class TestLucyImageFamily:
    def test_current_image_models_match(self) -> None:
        from genblaze_decart import DecartImageProvider

        provider = DecartImageProvider(api_key="test")
        for slug in ("lucy-pro-t2i", "lucy-pro-i2i"):
            match = provider._models.match_family(slug)
            assert match is not None, slug
            assert match.family.name == "decart-lucy-image", slug


class TestCrossModalityIsolation:
    """Video slugs must not match the image family and vice versa.
    The two providers ship separate registries with separate family
    patterns, but mismatches would cause subtle cost-tracking and
    metadata bugs."""

    def test_video_slugs_dont_match_image_family(self) -> None:
        from genblaze_decart import DecartImageProvider

        provider = DecartImageProvider(api_key="test")
        for slug in ("lucy-pro-t2v", "lucy-motion", "lucy-restyle-v2v"):
            assert provider._models.match_family(slug) is None, slug

    def test_image_slugs_dont_match_video_family(self) -> None:
        from genblaze_decart import DecartVideoProvider

        provider = DecartVideoProvider(api_key="test")
        for slug in ("lucy-pro-t2i", "lucy-pro-i2i"):
            assert provider._models.match_family(slug) is None, slug


# --- validate_model end-to-end --------------------------------------------


class TestValidateModel:
    def test_video_family_matched_provisional(self) -> None:
        from genblaze_decart import DecartVideoProvider

        provider = DecartVideoProvider(api_key="test")
        result = provider.validate_model("lucy-pro-t2v")
        assert result.outcome is ValidationOutcome.OK_PROVISIONAL
        assert result.family_name == "decart-lucy-video"

    def test_image_family_matched_provisional(self) -> None:
        from genblaze_decart import DecartImageProvider

        provider = DecartImageProvider(api_key="test")
        result = provider.validate_model("lucy-pro-t2i")
        assert result.outcome is ValidationOutcome.OK_PROVISIONAL
        assert result.family_name == "decart-lucy-image"

    def test_unmatched_slug_unknown_permissive(self) -> None:
        from genblaze_decart import DecartVideoProvider

        provider = DecartVideoProvider(api_key="test")
        result = provider.validate_model("not-a-decart-slug")
        assert result.outcome is ValidationOutcome.UNKNOWN_PERMISSIVE


# --- Pricing-removed contract --------------------------------------------


class TestPricingPhaseOut:
    def test_video_default_spec_no_pricing(self) -> None:
        from genblaze_decart import DecartVideoProvider

        provider = DecartVideoProvider(api_key="test")
        assert provider._models.get("lucy-pro-t2v").pricing is None

    def test_image_default_spec_no_pricing(self) -> None:
        from genblaze_decart import DecartImageProvider

        provider = DecartImageProvider(api_key="test")
        assert provider._models.get("lucy-pro-t2i").pricing is None
