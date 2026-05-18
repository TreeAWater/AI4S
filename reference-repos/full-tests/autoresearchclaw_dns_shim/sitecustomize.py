"""Test-only DNS shim for AutoResearchClaw SSRF unit tests.

The local execution environment resolves public test domains such as
example.com and arxiv.org to 198.18.0.0/15 benchmarking addresses. The project
SSRF guard correctly treats those as reserved/private, which makes two unit
tests environment-dependent. This shim is loaded only through PYTHONPATH for
the test process and leaves repository code unchanged.
"""

from __future__ import annotations

import socket

_real_getaddrinfo = socket.getaddrinfo
_public_overrides = {
    "example.com": "93.184.216.34",
    "arxiv.org": "151.101.67.42",
}


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host in _public_overrides:
        host = _public_overrides[host]
    return _real_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _patched_getaddrinfo
