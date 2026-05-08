"""Concurrency tests for ``BaseProvider._poll_cache``.

Pins the lock invariants that the v3 hardening introduced:

1. **No double-pop**: two concurrent callers of ``_get_cached_poll_result``
   on the same prediction_id — exactly one returns the cached result,
   the other returns ``None``.
2. **No torn reads**: ``_poll_cache`` and ``_poll_cache_times`` stay
   consistent across all observers. Either both have the entry or
   neither does.
3. **Cleanup races safely with reads/writes**: a stress test where
   8 threads cycle through cache/get/cleanup never raises
   ``RuntimeError: dictionary changed size during iteration``.

Run repeatedly with ``pytest --count=20`` (pytest-repeat) — race bugs
hide in flaky tests. A test that passes once and lies about safety
isn't a regression pin.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from genblaze_core.models.step import Step
from genblaze_core.providers.base import SyncProvider


class _PollProvider(SyncProvider):
    name = "poll-test"

    @classmethod
    def create_registry(cls):
        from genblaze_core.providers import ModelRegistry

        return ModelRegistry()

    def generate(self, step: Step, config=None) -> Step:  # pragma: no cover
        return step


def _make_provider() -> _PollProvider:
    return _PollProvider()


# ---------------------------------------------------------------------------
# Single-thread sanity (these would still pass without the lock)
# ---------------------------------------------------------------------------


def test_cache_and_consume() -> None:
    p = _make_provider()
    p._cache_poll_result("pred-1", {"status": "completed"})
    result = p._get_cached_poll_result("pred-1")
    assert result == {"status": "completed"}
    # Consumed — second read is None.
    assert p._get_cached_poll_result("pred-1") is None


def test_cleanup_removes_stale_entries() -> None:
    p = _make_provider()
    p._poll_cache_max_age = 0.05  # 50ms
    p._cache_poll_result("old", "x")
    time.sleep(0.1)
    p._cache_poll_result("fresh", "y")
    p._cleanup_poll_cache()
    assert p._get_cached_poll_result("old") is None
    assert p._get_cached_poll_result("fresh") == "y"


# ---------------------------------------------------------------------------
# Concurrency invariants
# ---------------------------------------------------------------------------


def test_no_double_pop_under_concurrent_get() -> None:
    """Two concurrent ``_get_cached_poll_result`` calls on the same
    key: exactly one wins the result, the other gets ``None``."""
    p = _make_provider()
    p._cache_poll_result("pred-1", "result-payload")

    barrier = threading.Barrier(2)
    results = []

    def consume() -> None:
        barrier.wait()  # release both threads simultaneously
        results.append(p._get_cached_poll_result("pred-1"))

    t1 = threading.Thread(target=consume)
    t2 = threading.Thread(target=consume)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one winner; exactly one None.
    assert sorted([r is None for r in results]) == [False, True]
    winners = [r for r in results if r is not None]
    assert winners == ["result-payload"]


def test_dict_consistency_under_concurrent_writes_and_reads() -> None:
    """``_poll_cache`` and ``_poll_cache_times`` must stay synchronized.
    Pop on one without the other under concurrent access would leave
    one dict ahead of the other. The lock prevents this."""
    p = _make_provider()
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def writer(idx: int) -> None:
        try:
            barrier.wait()
            for i in range(50):
                p._cache_poll_result(f"pred-{idx}-{i}", f"r-{idx}-{i}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader(idx: int) -> None:
        try:
            barrier.wait()
            for i in range(50):
                p._get_cached_poll_result(f"pred-{idx}-{i}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)] + [
        threading.Thread(target=reader, args=(i,)) for i in range(4)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"unexpected exceptions: {errors}"
    # Whatever's left in _poll_cache must have a corresponding entry in _poll_cache_times.
    assert set(p._poll_cache.keys()) == set(p._poll_cache_times.keys()), (
        "poll caches drifted out of sync"
    )


def test_cleanup_races_safely_with_writes() -> None:
    """Cleanup running concurrently with writes never raises
    ``RuntimeError: dictionary changed size during iteration``."""
    p = _make_provider()
    p._poll_cache_max_age = 0.001  # immediately stale
    barrier = threading.Barrier(9)
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        try:
            barrier.wait()
            i = 0
            while not stop.is_set():
                p._cache_poll_result(f"pred-{i}", "x")
                i += 1
                if i >= 1000:
                    break
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def cleaner() -> None:
        try:
            barrier.wait()
            for _ in range(500):
                p._cleanup_poll_cache()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(8)] + [
        threading.Thread(target=cleaner)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    stop.set()

    assert not errors, f"unexpected exceptions: {errors}"


def test_eight_thread_stress() -> None:
    """The acceptance criterion 12 stress: 8 threads × 100 cycles of
    write/read/cleanup. No exceptions; final state coherent."""
    p = _make_provider()
    p._poll_cache_max_age = 60.0  # don't auto-expire during the test

    def cycle(thread_idx: int) -> None:
        for i in range(100):
            key = f"t{thread_idx}-{i}"
            p._cache_poll_result(key, i)
            got = p._get_cached_poll_result(key)
            # Either we get the value we wrote, OR another thread
            # cycled the same key (won't happen with thread-prefixed keys).
            assert got == i, f"got {got} for {key}"
            if i % 10 == 0:
                p._cleanup_poll_cache()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(cycle, t) for t in range(8)]
        for fut in futures:
            fut.result(timeout=10.0)

    # Every thread fully consumed its entries.
    assert p._poll_cache == {}
    assert p._poll_cache_times == {}
