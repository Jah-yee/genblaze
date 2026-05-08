"""Conformance: no connector calls ``ModelRegistry(defaults=...)``.

The transitional ``defaults={}`` shim was removed in PR #13 of the
model-registry-decoupling rollout. This test pins that no connector
reintroduces it. Functionally redundant with the constructor's
runtime ``TypeError`` (``defaults=`` is no longer a parameter), but it
surfaces the violation as a clear, actionable failure at
``pytest`` rather than as an opaque import-time crash.

Connector authors migrating a registry should use either:

* ``ModelRegistry(provider_families=(...))`` for pattern-keyed routing
* ``reg = ModelRegistry(); reg.register(spec)`` for exact-match user specs
* ``reg.extend(specs)`` for bulk loading

See ``docs/exec-plans/active/model-registry-decoupling.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

# Match ``ModelRegistry(...defaults=...)`` across one or more lines so a
# connector can't sneak the kwarg back in via line-wrapping. The pattern
# anchors on the constructor name and a literal ``defaults=`` keyword;
# unrelated identifiers like ``param_defaults`` (a ``ModelSpec`` field)
# never have a leading word boundary that matches.
_DEFAULTS_KWARG = re.compile(
    r"\bModelRegistry\s*\([^)]*\bdefaults\s*=",
    re.MULTILINE | re.DOTALL,
)


def _connector_root() -> Path:
    """Return the absolute path to ``libs/connectors`` from this test file."""
    # tests/unit/test_no_defaults_kwarg.py → genblaze/libs/connectors
    return Path(__file__).resolve().parents[3] / "connectors"


def test_no_connector_calls_modelregistry_with_defaults_kwarg() -> None:
    """Every connector under ``libs/connectors/`` is forbidden from
    passing ``defaults=`` to ``ModelRegistry``."""
    root = _connector_root()
    assert root.is_dir(), f"Expected connectors directory at {root}"

    offenders: list[str] = []
    for py in root.rglob("*.py"):
        # Skip test fixtures — only production code is gated.
        if "tests" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        if _DEFAULTS_KWARG.search(text):
            offenders.append(str(py.relative_to(root.parent)))

    assert not offenders, (
        "ModelRegistry(defaults=...) is gone in 0.3.0. Migrate to "
        "provider_families=(...) or post-construction register()/extend(). "
        "Offending files:\n  - " + "\n  - ".join(offenders)
    )
