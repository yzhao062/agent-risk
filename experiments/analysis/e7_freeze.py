"""Write e7/SHA256SUMS (pre-registration docs/agent-risk-modeling/E7-PREREG-2026-10-06.md).

The list is every file E7 runs or imports: the E7 files below, plus every line of e1/SHA256SUMS, e2/SHA256SUMS
and e3/SHA256SUMS (E1's, E2's and E3's frozen files, unchanged, the D66 files they list, and E3's analysis
chain e3_analyze.py, e3_checker.py and e3/vendor/ that supply the primary intervals), deduplicated by path.
`run_e7.sh` refuses to start unless `sha256sum -c e7/SHA256SUMS` reads OK for every line on the host.
e7/SHA256SUMS does not list itself.

  python e7_freeze.py            # writes e7/SHA256SUMS and prints its sha256 and line count
"""

from __future__ import annotations

import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E7_FILES = [
    "agent_risk_harness/e7.py", "agent_risk_harness/e7_units.py", "make_manifest_e7.py", "run_e7.py", "run_e7.sh",
    "extract_e7.py", "analyze_e7.py", "e7_snapshots.py", "e7_hosts.py", "e7_provision.sh", "e7_recover.sh", "e7_watch.sh",
    "e7_freeze.py", "test_e7.py",
    "e7/manifest_e7.json", "e7/manifest_e7_smoke.json",
]


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(os.path.join(HERE, p), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    lines: dict[str, str] = {}
    for lst in ("e1/SHA256SUMS", "e2/SHA256SUMS", "e3/SHA256SUMS"):
        for line in open(os.path.join(HERE, lst)):
            if not line.strip():
                continue
            h, p = line.split(None, 1)
            p = p.strip().lstrip("*")
            if p.endswith("SHA256SUMS"):
                continue
            lines.setdefault(p, h)
    for p in E7_FILES:
        lines[p] = sha(p)
    out = os.path.join(HERE, "e7", "SHA256SUMS")
    with open(out, "w") as f:
        for p, h in lines.items():
            f.write(f"{h}  {p}\n")
    print(f"e7/SHA256SUMS: {len(lines)} lines, sha256 {sha('e7/SHA256SUMS')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
