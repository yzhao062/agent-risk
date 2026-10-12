"""Tests of e7_report_text.py on synthetic inputs (plan docs/agent-risk-modeling/PLAN-CORE-R7-2026-10-06.md,
"E7 Reporting Layer"). Reads no E7 record.

The base inputs come from analyze_e7.py's own synthetic world through predict and compare, so every JSON path
has its producer's shape; each case then sets only the fields it exercises. Importing the analyzer runs its
loaders, so run this where analyze_e7.py runs (numpy and scipy):

  python test_e7_report_text.py
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e7_report_text as rt  # noqa: E402

ae7 = rt.load_analyzer()
N = 1000
G1, G2, NG = "retry gate 1", "retry gate 2", rt.NO_GATE
TMP = tempfile.mkdtemp(prefix="e7rt_test_")


def _base() -> tuple[dict, dict]:
    root = os.path.join(TMP, "world")
    man, _, _ = ae7.synthetic_world(root)
    pred = ae7.predict(man, root, B=50)
    ana = ae7.compare(man, root, pred, 0.015, 60)
    pred, ana = json.loads(json.dumps(pred, default=str)), json.loads(json.dumps(ana, default=str))
    pred["manifest_sha256"] = ana["manifest_sha256"] = "0" * 64
    return ana, pred


BASE_A, BASE_P = _base()


def fresh() -> tuple[dict, dict]:
    return copy.deepcopy(BASE_A), copy.deepcopy(BASE_P)


_count = [0]


def run(ana: dict, pred: dict, registered: str | None = None, keep_prediction_sha: bool = False,
        sums: str | None = None, analyzer: str | None = None) -> dict:
    _count[0] += 1
    d = os.path.join(TMP, f"case{_count[0]}")
    os.makedirs(d)
    pp = os.path.join(d, "prediction_e7.json")
    with open(pp, "w") as f:
        json.dump(pred, f, indent=1)
    sha = rt.sha256_file(pp)
    ana = copy.deepcopy(ana)
    if not keep_prediction_sha:
        ana["prediction"] = {"path": pp, "sha256": sha}
    ap = os.path.join(d, "analysis_e7.json")
    with open(ap, "w") as f:
        json.dump(ana, f, indent=1)
    return rt.report(ap, pp, sha if registered is None else registered, sums or rt.SUMS, analyzer or rt.ANALYZER,
                     ae7=ae7)


def clause(rep: dict, cid: str) -> dict:
    hits = [c for c in rep["clauses"] if c["id"] == cid]
    assert len(hits) == 1, (cid, [c["id"] for c in rep["clauses"]])
    return hits[0]


def refuses(fn, needle: str) -> None:
    try:
        fn()
    except rt.Refusal as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"no refusal ({needle})")


# ------------------------------------------------------------ setters

def set_contrast(c: dict, nc90, nc95, cb90, cb95, D: float = 0.015) -> None:
    """Intervals plus the four flags, by analyze_e7.contrast's rules."""
    c["newcombe"]["newcombe_paired_90"], c["newcombe"]["newcombe_paired_95"] = nc90, nc95
    c["cluster"]["cluster_boot_90"], c["cluster"]["cluster_boot_95"] = cb90, cb95
    c["equivalent"] = bool(nc90[0] > -D and nc90[1] < D and cb90[0] > -D and cb90[1] < D)
    c["newcombe_90_inside"] = bool(nc90[0] > -D and nc90[1] < D)
    c["newcombe95_excludes_0"] = bool(nc95[0] > 0 or nc95[1] < 0)
    c["cluster95_excludes_0"] = bool(cb95[0] > 0 or cb95[1] < 0)


def set_primary(ana: dict, g: str, nc90, nc95, cb90, cb95) -> None:
    gp = ana["gates"][g]
    set_contrast(gp["primary"], nc90, nc95, cb90, cb95)
    om = ae7.outcome_map(int(g), gp["lam"], gp["primary"], gp["identified"]["p_harm"], gp["live"]["p_harm"], 5, 0.015)
    gp["outcome_map"] = json.loads(json.dumps(om))


def set_reference(ana: dict, g: str, key: str, nc95, cb95, eq: bool = False) -> None:
    r = ana["gates"][g]["references"][key]
    r["contrast"]["newcombe"]["newcombe_paired_95"] = nc95
    r["contrast"]["cluster"]["cluster_boot_95"] = cb95
    r["equivalent_to_live"] = eq


def make_refs(n: int, pa: tuple, st: tuple, unknown: int = 0) -> list[dict]:
    """pa, st: (harm count, success count); the last `unknown` units have unknown stop outcomes."""
    out = []
    for i in range(n):
        x = {"unit_id": f"u{i}", "b": 0, "pass_all_harm": int(i < pa[0]), "pass_all_failure": int(i >= pa[1]),
             "stop_harm": int(i < st[0]), "stop_failure": int(i >= st[1])}
        if i >= n - unknown:
            x["stop_harm"] = x["stop_failure"] = None
        out.append(x)
    return out


def set_decision(ana: dict, pred: dict, pass_all: tuple, ident: dict, live: dict, stop: dict, n: int = N,
                 unknown: int = 0) -> None:
    ana["n_units"] = pred["n_units"] = n
    pa = {"harm_k": pass_all[0], "succ_k": pass_all[1]}
    arm = lambda d: {g: {"harm_k": d[g][0], "succ_k": d[g][1]} for g in rt.GATES}  # noqa: E731
    ana["decision_example"] = json.loads(json.dumps(ae7.decision(arm(ident), arm(live), pa, n), default=str))
    for g in rt.GATES:
        pred["gates"][g]["refs"] = make_refs(n, pass_all, stop[g], unknown)


def decision_case(pass_all, ident, live=None, stop=None, unknown=0) -> dict:
    ana, pred = fresh()
    set_decision(ana, pred, pass_all, ident, live or ident, stop or ident, unknown=unknown)
    return run(ana, pred)


def rd(rep: dict, key: str) -> dict:
    return clause(rep, f"reference_decision.{key}")


# ------------------------------------------------------------ cases

def test_paths() -> None:
    """Every JSON path the plan names is read, and every clause names its gate."""
    rep = run(*fresh())
    a, p = set(rep["paths_read"]["analysis"]), set(rep["paths_read"]["prediction"])
    need_a = {"prediction.sha256", "analysis_sha256", "manifest_sha256", "delta", "n_units",
              "decision_example.harm_ceiling", "decision_example.success_floor",
              "decision_example.identified.candidates", "decision_example.live.candidates",
              "decision_example.identified.candidate", "decision_example.live.candidate",
              "baseline_collection.agent_calls", "baseline_collection.cost_recovery", "snapshot_cost"}
    need_p = {"analysis_sha256", "manifest_sha256", "n_units"}
    for g in rt.GATES:
        G = f"gates.{g}"
        need_a |= {f"{G}.outcome_map.branch", f"{G}.outcome_map.equivalent", f"{G}.outcome_map.newcombe95_excludes_0",
                   f"{G}.outcome_map.cluster95_excludes_0", f"{G}.identified.p_harm", f"{G}.live.p_harm",
                   f"{G}.identified.p_harm.unknown", f"{G}.identified.p_failure.unknown",
                   f"{G}.execution_cost.continuations.started", f"{G}.execution_cost.agent_calls_ratio",
                   f"{G}.execution_cost.continuations.complete", f"{G}.execution_cost.live.complete",
                   f"{G}.execution_cost.continuations.agent_calls", f"{G}.execution_cost.live.agent_calls",
                   f"{G}.budgeted_execution.curve.R_executable", f"{G}.budgeted_execution.matched_cost_note",
                   f"{G}.budgeted_execution.curve.smallest_k_certifying",
                   f"{G}.references.log_alone_bounds.p_harm_bounds",
                   f"{G}.pathwise.baseline_same_request_same_output.rate", f"{G}.pathwise.note",
                   f"{G}.secondary.failure.equivalent", f"{G}.sensitivity_S1.equivalence_survives_S1"}
        for st in ("b0", "b_gt_0"):
            need_a |= {f"{G}.pathwise.{st}.{k}" for k in ("next_turn_identical", "context_equal", "n_compared",
                                                           "trajectory_identical", "harm_equal")}
        for key in ("pass_all", "retry_as_stop"):
            need_a |= {f"{G}.references.{key}.equivalent_to_live",
                       f"{G}.references.{key}.contrast.newcombe.newcombe_paired_95",
                       f"{G}.references.{key}.contrast.cluster.cluster_boot_95"}
        need_p |= {f"{G}.refs", f"{G}.decomposition.continuing_mass", f"{G}.decomposition.continuations_valid",
                   f"{G}.decomposition.continuations_not_valid"}
    assert need_a <= a, sorted(need_a - a)
    assert need_p <= p, sorted(need_p - p)
    for c in rep["clauses"]:
        if c["id"].endswith((".g1", ".g2")):
            assert f"gate {c['id'][-1]}" in c["text"], c
        for bad in ("—", "–", "n't", "it's"):
            assert bad not in c["text"], (bad, c["id"])


def test_outcome_map_mixed() -> None:
    """Gate 1 equivalence_only and gate 2 exclusion_only in one analysis; the other two branches; no 'fail'."""
    ana, pred = fresh()
    set_primary(ana, "1", [-0.010, 0.012], [-0.012, 0.014], [-0.011, 0.013], [-0.013, 0.015])
    set_primary(ana, "2", [0.002, 0.030], [0.001, 0.033], [-0.001, 0.031], [-0.002, 0.034])
    rep = run(ana, pred)
    t1, t2 = clause(rep, "primary.g1")["text"], clause(rep, "primary.g2")["text"]
    assert clause(rep, "primary.g1")["fields"]["branch"] == "equivalence_only"
    assert "Marginal P(harm) is equivalent within 0.015 (both 90% intervals inside the margin" in t1
    assert "excludes zero" not in t1
    assert clause(rep, "primary.g2")["fields"]["branch"] == "exclusion_only"
    assert "Marginal P(harm) equivalence within 0.015 is not established" in t2
    assert "Newcombe 95% interval [0.0010, 0.0330] excludes zero" in t2 and "[-0.0020, 0.0340] contains zero" in t2
    assert "size and the cause of the discrepancy are not established" in t2
    ana, pred = fresh()
    set_primary(ana, "1", [0.001, 0.010], [0.0005, 0.012], [-0.001, 0.011], [-0.002, 0.013])
    set_primary(ana, "2", [-0.020, 0.010], [-0.025, 0.012], [-0.021, 0.011], [-0.026, 0.013])
    rep = run(ana, pred)
    t1, t2 = clause(rep, "primary.g1")["text"], clause(rep, "primary.g2")["text"]
    assert clause(rep, "primary.g1")["fields"]["branch"] == "equivalence_and_exclusion"
    assert "excludes zero (Newcombe only; the task-cluster 95% interval [-0.0020, 0.0130] contains zero)" in t1
    assert clause(rep, "primary.g2")["fields"]["branch"] == "neither"
    assert t2.split(". ", 1)[1].startswith("Neither marginal P(harm) equivalence within 0.015 nor a discrepancy")
    for c in rep["clauses"]:
        if c["id"].startswith("primary"):
            assert "fail" not in c["text"].lower(), c["text"]


def test_outcome_map_inconsistent_refuses() -> None:
    ana, pred = fresh()
    ana["gates"]["1"]["outcome_map"]["branch"] = "neither"
    ana["gates"]["1"]["outcome_map"]["equivalent"] = True
    refuses(lambda: run(ana, pred), "does not match its flags")


def test_envelope_capped() -> None:
    ana, pred = fresh()
    set_primary(ana, "1", [-1.0, 1.0], [-1.0, 1.0], [-0.01, 0.01], [-0.012, 0.012])
    ana["gates"]["1"]["primary"]["newcombe"]["envelope_capped"] = True
    rep = run(ana, pred)
    c = clause(rep, "primary.g1")
    assert c["fields"]["branch"] == "neither" and c["fields"]["envelope_capped"]
    assert "capped at [-1, 1]" in c["text"]


def test_references() -> None:
    """A reference separated for gate 2 only; separation on the Newcombe interval only; a touching boundary;
    equivalence; separation below."""
    ana, pred = fresh()
    set_reference(ana, "1", "retry_as_stop", [-0.010, 0.030], [-0.012, 0.031])
    set_reference(ana, "2", "retry_as_stop", [0.020, 0.060], [0.018, 0.070])
    set_reference(ana, "1", "pass_all", [0.020, 0.050], [0.010, 0.060])
    set_reference(ana, "2", "pass_all", [0.015, 0.050], [0.020, 0.060])
    rep = run(ana, pred)
    k = lambda cid: clause(rep, cid)["fields"]["kind"]  # noqa: E731
    assert k("reference.retry_as_stop.g1") == "not_established"
    assert k("reference.retry_as_stop.g2") == "outside"
    assert "marginal P(harm) is outside the margin (live above the reference" in clause(rep, "reference.retry_as_stop.g2")["text"]
    assert k("reference.pass_all.g1") == "newcombe_only"
    assert "(Newcombe interval only)" in clause(rep, "reference.pass_all.g1")["text"]
    assert k("reference.pass_all.g2") == "not_established"          # 0.015 touches the boundary: does not qualify
    assert clause(rep, "reference.pass_all.g2")["text"].count("not established at this margin;") == 1
    ana, pred = fresh()
    set_reference(ana, "1", "pass_all", [-0.010, 0.010], [-0.011, 0.012], eq=True)
    set_reference(ana, "2", "pass_all", [-0.060, -0.020], [-0.070, -0.016])
    rep = run(ana, pred)
    assert "marginal P(harm) is equivalent to live at the margin" in clause(rep, "reference.pass_all.g1")["text"]
    assert "live below the reference" in clause(rep, "reference.pass_all.g2")["text"]
    for c in rep["clauses"]:
        if c["id"].startswith("reference.") and "as good as" in c["text"]:
            raise AssertionError(c["text"])


def test_decisions() -> None:
    """Disagreeing decisions; neither arm certifies; both recommend no gate; both recommend the same gate."""
    rep = decision_case((300, 600), {"1": (100, 500), "2": (100, 450)}, live={"1": (300, 500), "2": (100, 450)})
    c = clause(rep, "decision")
    assert c["fields"]["kind"] == "differ"
    assert c["text"].startswith(f"The identified arm recommends {G1}; the live arm recommends {G2}")
    rep = decision_case((300, 600), {"1": (300, 600), "2": (300, 500)})
    assert clause(rep, "decision")["text"].startswith("Neither arm certifies a candidate at these requirements")
    rep = decision_case((100, 600), {"1": (100, 500), "2": (100, 450)})
    t = clause(rep, "decision")["text"]
    assert t.startswith("Both arms recommend no gate") and "shared pass-all evidence, so it is not agreement about a gate" in t
    rep = decision_case((300, 600), {"1": (100, 380), "2": (100, 500)})
    assert clause(rep, "decision")["text"].startswith(f"Both arms recommend {G2} (")
    b = clause(rep, "decision.bounds")
    assert b["fields"][f"identified:{G2}"]["harm_ub95_up"] >= ae7.cp_upper(100, N, 0.05)
    assert b["fields"][f"identified:{G2}"]["success_lb95_down"] <= ae7.cp_lower(500, N, 0.05)


def test_decision_tampered_refuses() -> None:
    ana, pred = fresh()
    set_decision(ana, pred, (300, 600), {"1": (100, 500), "2": (100, 450)}, {"1": (100, 500), "2": (100, 450)},
                 {"1": (100, 500), "2": (100, 450)})
    ana["decision_example"]["live"]["candidates"][1]["harm_ok"] = False
    refuses(lambda: run(ana, pred), "does not reproduce decision_example")


def test_ref_success_floor_only() -> None:
    rep = decision_case((300, 600), {"1": (100, 500), "2": (100, 300)}, stop={"1": (100, 300), "2": (100, 300)})
    f = rd(rep, "retry_as_stop")["fields"]["identified"]
    assert f["comparator_candidate"] == G1 and f["reference_candidate"] is None
    assert f["requirements_failed"] == "success" and not f["discriminates_on_harm"] and not f["harm_certification_differs"]
    assert f"under the reference counts {G1} fails the success floor." in rd(rep, "retry_as_stop")["text"]


def test_ref_ranking_only() -> None:
    rep = decision_case((300, 600), {"1": (100, 450), "2": (100, 500)}, stop={"1": (100, 550), "2": (100, 500)})
    f = rd(rep, "retry_as_stop")["fields"]["identified"]
    assert (f["comparator_candidate"], f["reference_candidate"]) == (G2, G1)
    assert f["requirements_failed"] == "neither" and f["decided_by"] == "success ranking" and not f["discriminates_on_harm"]
    assert f"meets both requirements, and the success ranking selects {G1}" in rd(rep, "retry_as_stop")["text"]


def test_ref_tiebreak_only() -> None:
    rep = decision_case((300, 600), {"1": (100, 500), "2": (120, 500)}, stop={"1": (150, 500), "2": (120, 500)})
    f = rd(rep, "retry_as_stop")["fields"]["identified"]
    assert (f["comparator_candidate"], f["reference_candidate"]) == (G1, G2)
    assert f["requirements_failed"] == "neither" and f["decided_by"] == "harm-bound tiebreak"


def test_ref_harm_and_success() -> None:
    rep = decision_case((300, 600), {"1": (100, 500), "2": (300, 300)}, stop={"1": (300, 300), "2": (100, 500)})
    f = rd(rep, "retry_as_stop")["fields"]["identified"]
    assert (f["comparator_candidate"], f["reference_candidate"]) == (G1, G2)
    assert f["requirements_failed"] == "both" and f["discriminates_on_harm"]
    assert f["harm_certification_differs"] == [G2]
    assert rep["harm_discrimination"] == ["ignore-the-gate against identified", "ignore-the-gate against live",
                                         "retry-as-stop against identified", "retry-as-stop against live"]
    t = rd(rep, "retry_as_stop")["text"]
    assert f"fails both the harm ceiling and the success floor" in t
    assert f"Harm certification differs for {G2} (identified arm: not met; retry-as-stop reference: met)." in t


def test_codex_round5_case() -> None:
    """Harm certification differs on a candidate neither arm recommends, while the recommendation changes through
    the success floor: the recommended candidate fails only the success floor; gate 1 is a separate fact."""
    rep = decision_case((100, 500), {"1": (400, 200), "2": (100, 700)}, stop={"1": (100, 200), "2": (100, 200)})
    f = rd(rep, "retry_as_stop")["fields"]["identified"]
    assert (f["comparator_candidate"], f["reference_candidate"]) == (G2, NG)
    assert f["requirements_failed"] == "success" and not f["discriminates_on_harm"]
    assert f["harm_certification_differs"] == [G1]
    assert rep["harm_discrimination"] == []
    t = rd(rep, "retry_as_stop")["text"]
    assert f"under the reference counts {G2} fails the success floor." in t
    assert f"Harm certification differs for {G1}" in t
    assert t.index(f"fails the success floor") < t.index(f"Harm certification differs for {G1}")


def test_codex_round6_case() -> None:
    """The comparator certifies nothing and the reference recommends a gate: no requirement classification and
    no harm discrimination; candidate-level facts are still reported."""
    rep = decision_case((300, 600), {"1": (300, 550), "2": (300, 500)}, stop={"1": (100, 450), "2": (100, 400)})
    c = rd(rep, "retry_as_stop")
    f = c["fields"]["identified"]
    assert f["comparator_candidate"] is None and f["reference_candidate"] == G1
    assert f["requirements_failed"] is None and f["decided_by"] is None and not f["discriminates_on_harm"]
    assert f"The identified arm certifies no candidate; the retry-as-stop reference recommends {G1}." in c["text"]
    assert f["reference_candidates"][G1]["harm_ub95_up"] == 0.1170
    assert f["reference_candidates"][G1]["success_lb95_down"] == 0.4237
    assert set(f["harm_certification_differs"]) == {G1, G2}
    assert rep["harm_discrimination"] == []


def test_ignore_arm_tie() -> None:
    """The ignore-the-gate arm: an exact three-way tie, so no gate when pass-all certifies and none otherwise."""
    rep = decision_case((100, 500), {"1": (100, 600), "2": (300, 300)})
    c = rd(rep, "pass_all")
    assert c["fields"]["degenerate"] and c["fields"]["reference_candidate"] == NG
    f = c["fields"]["identified"]
    assert (f["comparator_candidate"], f["requirements_failed"], f["decided_by"]) == (G1, "neither", "list order")
    assert f"so this difference shows that {G1} changes the declared decision" in c["text"]
    rep = decision_case((300, 600), {"1": (100, 500), "2": (300, 300)})
    c = rd(rep, "pass_all")
    assert c["fields"]["reference_candidate"] is None
    assert c["fields"]["identified"]["requirements_failed"] == "harm"
    rep = decision_case((100, 600), {"1": (100, 500), "2": (100, 450)})
    c = rd(rep, "pass_all")
    assert c["fields"]["reference_candidate"] == NG and not c["fields"]["identified"]["differs"]
    assert "so this agreement is not evidence about a gate" in c["text"]
    rep = decision_case((300, 600), {"1": (300, 500), "2": (300, 450)})
    c = rd(rep, "pass_all")
    assert c["fields"]["reference_candidate"] is None and "both certify no candidate" in c["text"]


def test_complete_arm_not_one_gate() -> None:
    """A one-gate substitution would mix reference and comparator evidence; the layer uses the complete arm."""
    rep = decision_case((300, 600), {"1": (100, 500), "2": (100, 450)}, stop={"1": (100, 300), "2": (100, 300)})
    assert rd(rep, "retry_as_stop")["fields"]["reference_candidate"] is None
    one = ae7.decision({"1": {"harm_k": 100, "succ_k": 300}, "2": {"harm_k": 100, "succ_k": 450}},
                       {"1": {"harm_k": 100, "succ_k": 300}, "2": {"harm_k": 100, "succ_k": 450}},
                       {"harm_k": 300, "succ_k": 600}, N)
    assert one["identified"]["candidate"] == G2


def test_ref_unknowns_unfavorable() -> None:
    rep = decision_case((300, 600), {"1": (100, 500), "2": (100, 450)}, stop={"1": (100, 600), "2": (100, 600)},
                        unknown=40)
    arm = rd(rep, "retry_as_stop")["fields"]["arm_counts"]
    assert arm["1"] == {"harm_k": 140, "succ_k": 600}, arm      # 40 unknowns counted harmful; none of them a success


def test_never_run() -> None:
    """Continuations never run: started below planned, unknown success above unknown harm, and the reason
    breakdown over planned continuing units."""
    ana, pred = fresh()
    n = ana["n_units"]
    pg = pred["gates"]["1"]["decomposition"]
    planned = int(round(pg["continuing_mass"] * n))
    pg["continuations_valid"] = planned - 2
    pg["continuations_not_valid"] = {"not run": 1, "no snapshot record at the first block": 1}
    ana["gates"]["1"]["execution_cost"]["continuations"]["started"] = planned - 2
    ana["gates"]["1"]["identified"]["p_harm"]["unknown"] = 1
    ana["gates"]["1"]["identified"]["p_failure"]["unknown"] = 2
    rep = run(ana, pred)
    m = clause(rep, "mass.g1")["text"]
    assert f"{planned - 2} valid of {planned - 2} started" in m
    assert f"({planned} of {n} planned units)" in m and "1 for harm, 2 for task success" in m
    r = clause(rep, "reasons.g1")["text"]
    assert f"reasons planned continuing units lack valid outcomes (denominator: {planned} planned continuing units)" in r
    assert "no snapshot record at the first block, 1; not run, 1" in r
    for bad in ("remainder", "of the started", "remaining"):
        assert bad not in m + r, bad


def test_status() -> None:
    ana, pred = fresh()
    ana["gates"]["1"]["references"]["log_alone_bounds"]["p_harm_bounds"] = [0.02275, 0.42125]
    ana["gates"]["2"]["references"]["log_alone_bounds"]["p_harm_bounds"] = [0.02, 0.24]
    rep = run(ana, pred)
    s1, s2 = clause(rep, "status.g1")["text"], clause(rep, "status.g2")["text"]
    assert "[0.022, 0.422] (plug-in outer interval). The log alone cannot approve the gate." in s1
    assert "cannot approve" not in s2


def test_cost_partial() -> None:
    ana, pred = fresh()
    ana["gates"]["1"]["execution_cost"]["live"]["complete"] = False
    ana["baseline_collection"]["cost_recovery"] = "partial"
    rep = run(ana, pred)
    assert clause(rep, "cost.given_log.g1")["fields"]["label"] == "partial"
    assert clause(rep, "cost.given_log.g2")["fields"]["label"] == "complete"
    assert clause(rep, "cost.log_charged.g2")["fields"]["label"] == "partial"
    assert "given the log (partial)" in clause(rep, "cost.given_log.g1")["text"]
    ec = ana["gates"]["2"]["execution_cost"]
    want = (ana["baseline_collection"]["agent_calls"] + ec["continuations"]["agent_calls"]) / ec["live"]["agent_calls"]
    assert abs(clause(rep, "cost.log_charged.g2")["fields"]["ratio"] - want) < 1e-12
    ana, pred = fresh()
    ana["baseline_collection"]["cost_recovery"] = "complete"
    rep = run(ana, pred)
    assert clause(rep, "cost.log_charged.g1")["fields"]["label"] == "complete"
    assert "charges the whole log to each gate" in clause(rep, "cost.log_charged.g1")["text"]


def test_pathwise_and_budget() -> None:
    rep = run(*fresh())
    t = clause(rep, "pathwise.g1")["text"]
    assert "Baseline, pooled over steps" in t and "context equal in" in t
    assert "Descriptive, not a test." in clause(rep, "pathwise_nobaseline.g1")["text"]
    b = clause(rep, "budget.g1")["text"]
    assert "the registered secondary analysis; bounds are pointwise" in b
    assert "not sequential certificates" in b and "certification-cost condition is unmet" in b
    assert "retrospective" in b
    ana, pred = fresh()
    be = ana["gates"]["1"]["budgeted_execution"]
    last = be["curve"]["grid"][-1]
    be["curve"]["smallest_k_certifying"]["0.25"] = last["k"]
    next(x for x in be["live_matched_cost"] if x["frac"] == last["frac"])["certify_share"]["0.25"] = 0.4
    b = clause(run(ana, pred), "budget.g1")["text"]
    R = be["curve"]["R_executable"]
    assert (f"0.25, {last['k']} of {R} continuations (the live gate at the same agent-call cost certifies it in 40% "
            f"of subsets)") in b, b


def test_refusals() -> None:
    ana, pred = fresh()
    refuses(lambda: run(ana, pred, registered="0" * 64), "does not equal the registered")
    a2 = copy.deepcopy(ana)
    a2["prediction"] = {"sha256": "f" * 64}
    refuses(lambda: run(a2, pred, keep_prediction_sha=True), "prediction.sha256")
    a2 = copy.deepcopy(ana)
    a2["delta"] = 0.02
    refuses(lambda: run(a2, pred), "margin")
    a2 = copy.deepcopy(ana)
    a2["decision_example"]["harm_ceiling"] = 0.2
    refuses(lambda: run(a2, pred), "harm ceiling")
    p2 = copy.deepcopy(pred)
    p2["manifest_sha256"] = "1" * 64
    refuses(lambda: run(ana, p2), "different manifests")
    p2 = copy.deepcopy(pred)
    p2["gates"]["2"]["refs"] = p2["gates"]["2"]["refs"][:-1]
    refuses(lambda: run(ana, p2), "one row per planned unit")
    sums = os.path.join(TMP, "SHA256SUMS")
    with open(sums, "w") as f:
        f.write("0" * 64 + "  analyze_e7.py\n")
    refuses(lambda: run(ana, pred, sums=sums), "SHA256SUMS")
    fake = os.path.join(TMP, "analyze_e7.py")
    with open(fake, "w") as f:
        f.write(open(rt.ANALYZER).read() + "\n# edited\n")
    refuses(lambda: run(ana, pred, analyzer=fake), "imported analyzer")


def test_cli() -> None:
    ana, pred = fresh()
    d = os.path.join(TMP, "cli")
    os.makedirs(d)
    pp, ap, out = os.path.join(d, "p.json"), os.path.join(d, "a.json"), os.path.join(d, "out.json")
    json.dump(pred, open(pp, "w"))
    ana["prediction"] = {"path": pp, "sha256": rt.sha256_file(pp)}
    json.dump(ana, open(ap, "w"))
    assert rt.main(["--analysis", ap, "--prediction", pp, "--registered-prediction-sha256", "0" * 64]) == 2
    assert rt.main(["--analysis", ap, "--prediction", pp, "--registered-prediction-sha256", rt.sha256_file(pp),
                    "--out", out]) == 0
    assert json.load(open(out))["schema"] == "e7-report-text-v1"


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    try:
        for t in tests:
            try:
                t()
                print(f"PASS {t.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL {t.__name__}")
                traceback.print_exc()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
