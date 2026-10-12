"""In-process egress guard for E3 (defense in depth beside the host rule).

``install()`` replaces socket connect and name resolution so that any attempt
to reach a non-loopback address raises ``EgressBlocked`` and is counted. The E3
runner installs it before importing tau2 (and litellm, which tau2 imports), so
an accidental call to a hosted API fails at once instead of leaving the host.
The host-level rule (``e3_net_block.sh``) is the registered egress block; this
guard only makes violations loud and testable on a machine without that rule.
"""

from __future__ import annotations

import ipaddress
import socket

ATTEMPTS: list[str] = []
_ORIG = {}


class EgressBlocked(OSError):
    pass


def _is_loopback_host(host) -> bool:
    if host is None:
        return False
    h = str(host)
    if h in ("localhost", "ip6-localhost", ""):
        return True
    try:
        return ipaddress.ip_address(h.split("%")[0]).is_loopback
    except ValueError:
        return False


def install() -> None:
    if _ORIG:
        return
    _ORIG["connect"] = socket.socket.connect
    _ORIG["connect_ex"] = socket.socket.connect_ex
    _ORIG["getaddrinfo"] = socket.getaddrinfo

    def _check(addr):
        if isinstance(addr, tuple) and addr and not _is_loopback_host(addr[0]):
            ATTEMPTS.append(str(addr[0]))
            raise EgressBlocked(f"E3 egress guard: connection to {addr[0]} refused")

    def connect(self, addr):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check(addr)
        return _ORIG["connect"](self, addr)

    def connect_ex(self, addr):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check(addr)
        return _ORIG["connect_ex"](self, addr)

    def getaddrinfo(host, *a, **k):
        if not _is_loopback_host(host):
            ATTEMPTS.append(f"dns:{host}")
            raise EgressBlocked(f"E3 egress guard: name resolution of {host} refused")
        return _ORIG["getaddrinfo"](host, *a, **k)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo


def uninstall() -> None:
    if not _ORIG:
        return
    socket.socket.connect = _ORIG.pop("connect")
    socket.socket.connect_ex = _ORIG.pop("connect_ex")
    socket.getaddrinfo = _ORIG.pop("getaddrinfo")
