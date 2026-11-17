"""Shared test setup: no real DNS. MX lookups answer "unknown" (as without dnspython) unless a test scripts them."""

import pytest


@pytest.fixture(autouse=True)
def _no_dns(monkeypatch):
    import brain.emailfinder as ef

    ef._mx_cache.clear()
    try:
        import dns.resolver

        def offline(*a, **k):
            raise RuntimeError("DNS disabled in tests")

        monkeypatch.setattr(dns.resolver, "resolve", offline)
    except ImportError:
        pass
    yield
    ef._mx_cache.clear()
