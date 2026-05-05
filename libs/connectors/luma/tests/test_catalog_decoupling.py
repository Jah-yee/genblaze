"""Catalog-decoupling tests for genblaze-luma 0.3.0.

Coverage:

* ``DiscoverySupport.NONE`` declared on LumaProvider.
* Family-pattern resolution: current ``ray-2`` / ``ray-flash-2``
  match, future ``ray-N`` variants inherit, non-Luma slugs fall
  through to the permissive fallback.
* Pricing-removed contract: registry default specs carry no pricing
  (Luma has always been pricing-None; the test pins the contract).
* User-registered pricing flows through ``compute_cost`` —
  documents the recipe shape for downstream callers.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from genblaze_core.models.step import Step
from genblaze_core.providers import (
    DiscoverySupport,
    PricingContext,
    PricingStrategy,
)


@pytest.fixture(autouse=True)
def _patch_luma_sdk():
    """Avoid importing the real lumaai package in tests."""
    with patch.dict("sys.modules", {"lumaai": MagicMock()}):
        yield


# --- DiscoverySupport declaration -----------------------------------------


class TestDiscoverySupportDeclaration:
    def test_none(self) -> None:
        from genblaze_luma import LumaProvider

        assert LumaProvider.discovery_support is DiscoverySupport.NONE


# --- Family resolution ----------------------------------------------------


class TestRayFamily:
    def test_current_models_match(self) -> None:
        from genblaze_luma import LumaProvider

        provider = LumaProvider(auth_token="test")
        for slug in ("ray-2", "ray-flash-2"):
            match = provider._models.match_family(slug)
            assert match is not None and match.family.name == "luma-ray", slug

    def test_future_variants_inherit(self) -> None:
        from genblaze_luma import LumaProvider

        provider = LumaProvider(auth_token="test")
        for slug in ("ray-3", "ray-pro-2", "ray-flash-3"):
            assert provider._models.match_family(slug) is not None, slug

    def test_non_ray_slugs_dont_match(self) -> None:
        from genblaze_luma import LumaProvider

        provider = LumaProvider(auth_token="test")
        for slug in ("dream-machine-1", "veo-3.0-generate-001", "sora-2"):
            assert provider._models.match_family(slug) is None, slug


# --- Pricing-removed contract --------------------------------------------


class TestPricingPhaseOut:
    def test_default_spec_no_pricing(self) -> None:
        from genblaze_luma import LumaProvider

        provider = LumaProvider(auth_token="test")
        for slug in ("ray-2", "ray-flash-2", "ray-3"):
            assert provider._models.get(slug).pricing is None, slug

    def test_user_registered_per_second_flows_through(self) -> None:
        """Document the canonical Luma pricing recipe."""
        from genblaze_luma import LumaProvider

        def per_second_by_model(rate: float) -> PricingStrategy:
            def _strategy(ctx: PricingContext) -> float | None:
                raw = ctx.step.params.get("duration")
                # Accept "5s" / "5" / 5 — strip trailing 's' if present.
                if isinstance(raw, str) and raw.endswith("s"):
                    raw = raw[:-1]
                try:
                    dur = int(raw) if raw is not None else 5
                except (TypeError, ValueError):
                    dur = 5
                count = ctx.output_count or 1
                return rate * dur * count

            return _strategy

        mock_client = MagicMock()
        mock_client.generations.get.return_value = SimpleNamespace(
            id="gen-1",
            state="completed",
            assets=SimpleNamespace(video="https://luma-output.com/v.mp4"),
        )
        provider = LumaProvider(auth_token="test")
        provider._client = mock_client
        # Fork before mutating to keep the class-level default registry clean.
        provider._models = provider.models.fork()
        provider.models.register_pricing("ray-2", per_second_by_model(0.40))

        step = Step(provider="luma", model="ray-2", prompt="x", params={"duration": "5s"})
        result = provider.fetch_output("gen-1", step)
        assert result.cost_usd == pytest.approx(0.40 * 5)
