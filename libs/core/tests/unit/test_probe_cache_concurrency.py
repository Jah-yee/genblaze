"""Concurrency + cleanup tests for ``BaseProvider._cached_probe``.

Pins three regression cases that the v3 hardening targeted:

1. **Exception cleanup** — when ``_invoke_family_probe`` raises an
   ``Exception``, ``_probe_inflight`` is empty after the call so a
   subsequent caller doesn't become a stuck waiter on a dead Event.
2. **BaseException cleanup** — same invariant under ``BaseException``
   (``KeyboardInterrupt``, ``SystemExit``, custom ``BaseException``
   subclasses). The original code path lacked ``try/finally`` and
   leaked the in-flight entry permanently.
3. **Concurrent waiters under hung probe** — the elected fetcher's
   probe blocks; waiters fall through to ``UNKNOWN`` after the timeout
   without permanently corrupting state.

Run repeatedly with ``pytest --count=20`` (pytest-repeat) to flush
out timing-dependent races; flaky once-passing tests don't catch
regressions.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from genblaze_core.models.enums import Modality
from genblaze_core.models.step import Step
from genblaze_core.providers import (
    DiscoverySupport,
    LiveProbeResult,
    ModelFamily,
    ModelRegistry,
    ModelSpec,
)
from genblaze_core.providers.base import SyncProvider


def _noop_probe(slug, **kwargs):
    return LiveProbeResult.LIVE


_TEST_FAMILY = ModelFamily(
    name="probe-test",
    pattern=re.compile(r"^probe-test-"),
    spec_template=ModelSpec(model_id="*", modality=Modality.IMAGE),
    description="Test family for probe-cache concurrency tests.",
    example_slugs=("probe-test-1",),
    probe=_noop_probe,
)


class _ProbeTestProvider(SyncProvider):
    """Minimal PARTIAL provider for exercising ``_cached_probe`` paths.

    ``_invoke_family_probe`` defaults to returning ``LIVE``; tests
    override via ``patch.object`` per case.
    """

    name = "probe-test"
    discovery_support = DiscoverySupport.PARTIAL

    @classmethod
    def create_registry(cls) -> ModelRegistry:
        return ModelRegistry(provider_families=(_TEST_FAMILY,))

    def _invoke_family_probe(self, probe, model_id):  # type: ignore[override]
        return probe(model_id)

    def generate(self, step: Step, config=None) -> Step:  # pragma: no cover
        return step


def _make_provider() -> _ProbeTestProvider:
    return _ProbeTestProvider()


# ---------------------------------------------------------------------------
# Cleanup invariants
# ---------------------------------------------------------------------------


def test_exception_in_probe_clears_inflight_entry() -> None:
    """A regular ``Exception`` from ``_invoke_family_probe`` must leave
    ``_probe_inflight`` empty so subsequent calls aren't stuck waiters."""
    provider = _make_provider()
    with patch.object(
        _ProbeTestProvider,
        "_invoke_family_probe",
        side_effect=RuntimeError("boom"),
    ):
        result = provider.validate_model("probe-test-1")
    # UNKNOWN is the framework's fail-open response; the assertion that
    # matters is the cleanup invariant below.
    assert result.outcome.value in {"ok_provisional", "unknown_permissive"}
    assert provider._probe_inflight == {}, "in-flight Event leaked after Exception in probe"


class _BoomBase(BaseException):
    """Custom BaseException subclass for test purposes — does NOT
    inherit from Exception, so the framework's ``except Exception``
    clause does not catch it. Only ``finally`` cleanup runs."""


def test_base_exception_in_probe_still_clears_inflight_entry() -> None:
    """The BLOCKER case from the v3 hardening review: ``BaseException``
    must run ``finally`` cleanup so the in-flight Event doesn't leak."""
    provider = _make_provider()
    with patch.object(
        _ProbeTestProvider,
        "_invoke_family_probe",
        side_effect=_BoomBase("interrupt"),
    ):
        with pytest.raises(_BoomBase):
            provider.validate_model("probe-test-1")
    # The exception propagated, AND the inflight slot was cleaned.
    assert provider._probe_inflight == {}, (
        "in-flight Event leaked after BaseException in probe — "
        "the original bug this test pins against"
    )


def test_repeated_base_exceptions_do_not_accumulate_inflight_entries() -> None:
    """Ten consecutive BaseException-raising probes must leave
    ``_probe_inflight`` empty — pin against gradual leak."""
    provider = _make_provider()
    with patch.object(
        _ProbeTestProvider,
        "_invoke_family_probe",
        side_effect=_BoomBase("interrupt"),
    ):
        for _ in range(10):
            with pytest.raises(_BoomBase):
                provider.validate_model("probe-test-1", refresh=True)
    assert len(provider._probe_inflight) == 0


def test_unknown_result_does_not_pollute_cache() -> None:
    """Phase-3 invariant: ``UNKNOWN`` results are NOT written to the
    probe cache. Transient failures shouldn't poison the TTL window."""
    provider = _make_provider()
    with patch.object(
        _ProbeTestProvider,
        "_invoke_family_probe",
        return_value=LiveProbeResult.UNKNOWN,
    ):
        provider.validate_model("probe-test-1")
    assert provider._probe_cache == {}, "UNKNOWN was cached — should be re-probed on next call"


def test_definitive_result_is_cached() -> None:
    """LIVE / DEAD results ARE cached for the TTL window."""
    provider = _make_provider()
    with patch.object(
        _ProbeTestProvider,
        "_invoke_family_probe",
        return_value=LiveProbeResult.LIVE,
    ):
        provider.validate_model("probe-test-1")
    assert "probe-test-1" in provider._probe_cache


# ---------------------------------------------------------------------------
# Concurrent waiters
# ---------------------------------------------------------------------------


def test_concurrent_callers_share_one_probe() -> None:
    """N concurrent callers for the same slug fire exactly one probe;
    the rest are waiters that read the result the elected fetcher
    produced."""
    provider = _make_provider()

    call_count = 0
    call_count_lock = threading.Lock()
    fetcher_started = threading.Event()
    fetcher_release = threading.Event()

    def slow_probe(probe, model_id):
        # patch.object replaces the unbound method, so side_effect
        # receives (probe, model_id) — matches `_invoke_family_probe`
        # signature without `self`.
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        fetcher_started.set()
        # Block until the test signals the fetcher to complete; this
        # forces the other callers into the waiter branch.
        fetcher_release.wait(timeout=5.0)
        return LiveProbeResult.LIVE

    with patch.object(_ProbeTestProvider, "_invoke_family_probe", side_effect=slow_probe):
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(provider.validate_model, "probe-test-1") for _ in range(8)]
            # Wait for the elected fetcher to enter the probe.
            assert fetcher_started.wait(timeout=5.0)
            # Now release; all 8 callers should resolve.
            fetcher_release.set()
            for fut in futures:
                fut.result(timeout=5.0)

    # Exactly one probe call across all 8 concurrent callers.
    assert call_count == 1, f"expected 1 probe, got {call_count}"
    assert provider._probe_inflight == {}


def test_concurrent_callers_during_hung_probe() -> None:
    """If the elected fetcher's probe never completes (within reason),
    waiters time out via ``_PROBE_INFLIGHT_WAIT_SECONDS`` and fall
    through to ``UNKNOWN`` rather than blocking forever."""
    provider = _make_provider()
    # Tighten the waiter timeout for this test.
    provider._PROBE_INFLIGHT_WAIT_SECONDS = 0.5  # type: ignore[misc]

    fetcher_started = threading.Event()
    never_release = threading.Event()  # never set

    def hung_probe(probe, model_id):
        fetcher_started.set()
        never_release.wait(timeout=2.0)  # eventually returns
        return LiveProbeResult.LIVE

    with patch.object(_ProbeTestProvider, "_invoke_family_probe", side_effect=hung_probe):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(provider.validate_model, "probe-test-1") for _ in range(4)]
            assert fetcher_started.wait(timeout=3.0)
            # Waiters should have given up by now and returned UNKNOWN-derived
            # OK_PROVISIONAL outcomes; the hung fetcher eventually cleans up.
            for fut in futures:
                result = fut.result(timeout=5.0)
                # All callers get a defined result (no exceptions, no hang).
                assert result.outcome is not None

    # State is clean after everything settles.
    assert provider._probe_inflight == {}
