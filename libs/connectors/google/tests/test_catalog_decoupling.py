"""Catalog-decoupling tests for genblaze-google 0.3.0.

Coverage:

* ``DiscoverySupport.PARTIAL`` declared on Veo + Imagen.
* Family-pattern resolution, with cross-modality isolation (Veo
  doesn't match imagen-/gemini- slugs and vice-versa).
* ``client.models.get`` family probe maps 200 → LIVE, 404 → DEAD,
  other errors → UNKNOWN.
* ``validate_model`` outcomes: OK_AUTHORITATIVE for live, NOT_FOUND
  for dead, OK_PROVISIONAL when probe inconclusive.
* Pricing-removed contract: registry default specs carry no pricing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from genblaze_core.providers import (
    DiscoverySupport,
    LiveProbeResult,
    ValidationOutcome,
    ValidationSource,
)


@pytest.fixture(autouse=True)
def _patch_google_sdk():
    """Avoid importing the real google-genai package in tests."""
    mock_types = MagicMock()
    mock_genai = MagicMock()
    mock_google_mod = MagicMock()
    mock_google_mod.genai = mock_genai
    with patch.dict(
        "sys.modules",
        {
            "google": mock_google_mod,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        },
    ):
        yield


# --- DiscoverySupport declarations ---------------------------------------


class TestDiscoverySupportDeclarations:
    def test_veo_partial(self) -> None:
        from genblaze_google import VeoProvider

        assert VeoProvider.discovery_support is DiscoverySupport.PARTIAL

    def test_imagen_partial(self) -> None:
        from genblaze_google import ImagenProvider

        assert ImagenProvider.discovery_support is DiscoverySupport.PARTIAL


# --- Family resolution + cross-modality isolation ------------------------


class TestVeoFamily:
    def test_current_models_match(self) -> None:
        from genblaze_google import VeoProvider

        provider = VeoProvider(api_key="test")
        for slug in (
            "veo-2.0-generate-001",
            "veo-3.0-generate-001",
            "veo-3.0-fast-generate-001",
        ):
            match = provider._models.match_family(slug)
            assert match is not None and match.family.name == "google-veo", slug

    def test_future_variants_inherit(self) -> None:
        from genblaze_google import VeoProvider

        provider = VeoProvider(api_key="test")
        for slug in ("veo-4.0-generate-001", "veo-3.0-ultra-generate-002"):
            assert provider._models.match_family(slug) is not None, slug

    def test_imagen_and_gemini_slugs_dont_match(self) -> None:
        from genblaze_google import VeoProvider

        provider = VeoProvider(api_key="test")
        for slug in ("imagen-3.0-generate-002", "gemini-2.5-flash"):
            assert provider._models.match_family(slug) is None, slug


class TestImagenFamily:
    def test_current_models_match(self) -> None:
        from genblaze_google import ImagenProvider

        provider = ImagenProvider(api_key="test")
        for slug in ("imagen-3.0-generate-002", "imagen-3.0-fast-generate-001"):
            match = provider._models.match_family(slug)
            assert match is not None and match.family.name == "google-imagen", slug

    def test_future_variants_inherit(self) -> None:
        from genblaze_google import ImagenProvider

        provider = ImagenProvider(api_key="test")
        for slug in ("imagen-4.0-generate-001", "imagen-3.0-ultra-001"):
            assert provider._models.match_family(slug) is not None, slug

    def test_veo_and_gemini_slugs_dont_match(self) -> None:
        from genblaze_google import ImagenProvider

        provider = ImagenProvider(api_key="test")
        for slug in ("veo-2.0-generate-001", "gemini-2.5-flash"):
            assert provider._models.match_family(slug) is None, slug


# --- Family probe (client.models.get) ------------------------------------


class TestProbeMapping:
    def test_returns_live_when_get_succeeds(self) -> None:
        from genblaze_google._probe import google_models_get_probe

        client = MagicMock()
        client.models.get.return_value = MagicMock(name="veo-3.0-generate-001")
        assert (
            google_models_get_probe("veo-3.0-generate-001", client=client) is LiveProbeResult.LIVE
        )
        client.models.get.assert_called_once_with(model="veo-3.0-generate-001")

    def test_returns_dead_on_404_status_attr(self) -> None:
        from genblaze_google._probe import google_models_get_probe

        client = MagicMock()
        err = Exception("model not found")
        err.status_code = 404
        client.models.get.side_effect = err
        assert google_models_get_probe("dead-slug", client=client) is LiveProbeResult.DEAD

    def test_returns_dead_on_404_in_message(self) -> None:
        from genblaze_google._probe import google_models_get_probe

        client = MagicMock()
        client.models.get.side_effect = Exception("404 NOT_FOUND: model not available")
        assert google_models_get_probe("dead-slug", client=client) is LiveProbeResult.DEAD

    def test_returns_unknown_on_403(self) -> None:
        from genblaze_google._probe import google_models_get_probe

        client = MagicMock()
        err = Exception("permission denied")
        err.status_code = 403
        client.models.get.side_effect = err
        assert google_models_get_probe("locked-slug", client=client) is LiveProbeResult.UNKNOWN

    def test_returns_unknown_on_transport_error(self) -> None:
        from genblaze_google._probe import google_models_get_probe

        client = MagicMock()
        client.models.get.side_effect = ConnectionError("dns failure")
        assert google_models_get_probe("any", client=client) is LiveProbeResult.UNKNOWN


# --- validate_model outcomes ---------------------------------------------


class TestValidateModelVeo:
    def test_authoritative_when_probe_live(self) -> None:
        from genblaze_google import VeoProvider

        provider = VeoProvider(api_key="test")
        client = MagicMock()
        client.models.get.return_value = MagicMock()
        provider._client = client

        result = provider.validate_model("veo-3.0-generate-001")
        assert result.outcome is ValidationOutcome.OK_AUTHORITATIVE
        assert result.source is ValidationSource.PROBE

    def test_not_found_when_probe_dead(self) -> None:
        from genblaze_google import VeoProvider

        provider = VeoProvider(api_key="test")
        client = MagicMock()
        err = Exception("404 not found")
        err.status_code = 404
        client.models.get.side_effect = err
        provider._client = client

        result = provider.validate_model("veo-9.9-ghost")
        assert result.outcome is ValidationOutcome.NOT_FOUND

    def test_provisional_when_probe_unknown(self) -> None:
        from genblaze_google import VeoProvider

        provider = VeoProvider(api_key="test")
        client = MagicMock()
        client.models.get.side_effect = ConnectionError("network down")
        provider._client = client

        result = provider.validate_model("veo-3.0-generate-001")
        assert result.outcome is ValidationOutcome.OK_PROVISIONAL


class TestValidateModelImagen:
    def test_authoritative_when_probe_live(self) -> None:
        from genblaze_google import ImagenProvider

        provider = ImagenProvider(api_key="test")
        client = MagicMock()
        client.models.get.return_value = MagicMock()
        provider._client = client

        result = provider.validate_model("imagen-3.0-generate-002")
        assert result.outcome is ValidationOutcome.OK_AUTHORITATIVE
        assert result.source is ValidationSource.PROBE


# --- Pricing-removed contract --------------------------------------------


class TestPricingPhaseOut:
    def test_veo_default_spec_no_pricing(self) -> None:
        from genblaze_google import VeoProvider

        provider = VeoProvider(api_key="test")
        for slug in (
            "veo-2.0-generate-001",
            "veo-3.0-generate-001",
            "veo-3.0-fast-generate-001",
        ):
            assert provider._models.get(slug).pricing is None, slug

    def test_imagen_default_spec_no_pricing(self) -> None:
        from genblaze_google import ImagenProvider

        provider = ImagenProvider(api_key="test")
        for slug in ("imagen-3.0-generate-002", "imagen-3.0-fast-generate-001"):
            assert provider._models.get(slug).pricing is None, slug
