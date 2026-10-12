"""Inter-coder agreement analysis for agent characteristics survey.

Reads TSV coding matrices from coding/ and computes agreement, Cohen's kappa,
per-code distributions, and summary counts for reporting.
"""

import collections
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODING_DIR = os.path.join(BASE_DIR, "coding")
CODES = [f"C{i}" for i in range(1, 21)]

FOCUS_CODES = [
    ("C2", "Weak or missing approval (C2)"),
    ("C3", "Irreversible actions (C3)"),
    ("C5", "Long horizons (C5)"),
    ("C7", "Persistent memory or state (C7)"),
    ("C8", "Multi-agent interaction (C8)"),
    ("C9", "Observability gaps (C9)"),
    ("C11", "Evaluation awareness (C11)"),
    ("C12", "Open versus closed access (C12)"),
    ("C13", "Model updates (C13)"),
]


def load_matrix(filename):
    path = os.path.join(CODING_DIR, filename)
    rows = []
    header = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cells = line.split("\t")
            if header is None:
                header = cells
                continue
            row_dict = dict(zip(header, cells))
            rows.append(row_dict)
    return rows


def norm(v):
    v = v.upper().strip()
    return v[:1] if v[:1] in "TNY0" else v


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    include_counts = "--no-counts" not in sys.argv

    # Load sources
    src1_raw = load_matrix("sources-coder1.tsv")
    src2_raw = load_matrix("sources-coder2.tsv")
    src1 = [[norm(r[c]) for c in CODES] for r in src1_raw]
    src2 = [[norm(r[c]) for c in CODES] for r in src2_raw]

    # Load incidents
    inc1_raw = load_matrix("incidents-coder1.tsv")
    inc2_raw = load_matrix("incidents-coder2.tsv")
    inc1 = [[norm(r[c]) for c in CODES] for r in inc1_raw]
    inc2 = [[norm(r[c]) for c in CODES] for r in inc2_raw]

    # Sources agreement
    flat1 = [v for row in src1 for v in row]
    flat2 = [v for row in src2 for v in row]
    po, k = kappa(flat1, flat2)
    print(f"sources 3-level T/N/0: agreement {po:.3f}, kappa {k:.3f}, cells {len(flat1)}")
    any1 = ["1" if v in "TN" else "0" for v in flat1]
    any2 = ["1" if v in "TN" else "0" for v in flat2]
    po, k = kappa(any1, any2)
    print(f"sources any mention (T or N vs 0): agreement {po:.3f}, kappa {k:.3f}")
    t1 = ["1" if v == "T" else "0" for v in flat1]
    t2 = ["1" if v == "T" else "0" for v in flat2]
    po, k = kappa(t1, t2)
    print(f"sources top level (T vs not): agreement {po:.3f}, kappa {k:.3f}")
    print("code | T coder1 | T coder2 | T both | T-or-N coder1 | coder2 | both | kappa(any)")

    src_counts = {}
    for j, c in enumerate(CODES):
        a = [r[j] for r in src1]
        b = [r[j] for r in src2]
        ka = kappa(["1" if v in "TN" else "0" for v in a], ["1" if v in "TN" else "0" for v in b])[1]
        t_c1 = sum(v == "T" for v in a)
        t_c2 = sum(v == "T" for v in b)
        t_both = sum(x == "T" and y == "T" for x, y in zip(a, b))
        any_c1 = sum(v in "TN" for v in a)
        any_c2 = sum(v in "TN" for v in b)
        any_both = sum(x in "TN" and y in "TN" for x, y in zip(a, b))
        src_counts[c] = (any_c1, any_c2, any_both, t_c1, t_c2, t_both)
        print(c, t_c1, t_c2, t_both, any_c1, any_c2, any_both, f"{ka:.2f}")

    # Incidents agreement
    flat1_inc = [v for row in inc1 for v in row]
    flat2_inc = [v for row in inc2 for v in row]
    po, k = kappa(flat1_inc, flat2_inc)
    print(f"incidents Y/0: agreement {po:.3f}, kappa {k:.3f}, cells {len(flat1_inc)}")
    print("code | Y coder1 | Y coder2 | Y both | kappa")

    inc_counts = {}
    for j, c in enumerate(CODES):
        a = [r[j] for r in inc1]
        b = [r[j] for r in inc2]
        y_c1 = sum(v == "Y" for v in a)
        y_c2 = sum(v == "Y" for v in b)
        y_both = sum(x == "Y" and y == "Y" for x, y in zip(a, b))
        inc_counts[c] = (y_c1, y_c2, y_both)
        print(c, y_c1, y_c2, y_both, f"{kappa(a, b)[1]:.2f}")

    if include_counts:
        print("\nCounts for Reporting:")
        print("Characteristic (code) | Sources naming it (coder 1 / coder 2 / both) | Top level (1 / 2 / both) | Incidents (1 / 2 / both)")
        for c, name in FOCUS_CODES:
            s_any_c1, s_any_c2, s_any_both, s_t_c1, s_t_c2, s_t_both = src_counts[c]
            i_y_c1, i_y_c2, i_y_both = inc_counts[c]
            print(f"{name} | {s_any_c1} / {s_any_c2} / {s_any_both} | {s_t_c1} / {s_t_c2} / {s_t_both} | {i_y_c1} / {i_y_c2} / {i_y_both}")


if __name__ == "__main__":
    main()
