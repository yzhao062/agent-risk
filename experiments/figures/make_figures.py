"""Figures D.1, D.2, and D.3 (implementation checks) from frozen analysis outputs.

Every plotted number is read from a released results file under data/logs/
(or the folder named by AGENT_RISK_LOGS; paths below) or computed here from counts in
those files; nothing is typed in by hand except the arm E counts, which are the
agreement table of registrations/ARM-E-2026-10-04.md
(data/arm-e/results/analysis/analysis.json) and are re-derived as exact
Clopper-Pearson intervals below. The registered relay margin 0.03 is also
entered as a fixed design parameter.

House palette: gray C9C9C9, mint BFDFD2 (identified), coral ED8D5A (required
additional evidence).

Two retry gates (Figures D.1 and D.2): logs/e7/analysis_e7.json (the registered, hash-guarded compare
output) and logs/e7/prediction_e7.json (frozen before any live record was read).
Two-agent relay (Figure D.1): logs/d66/analysis_d66.json.

Usage: python experiments/figures/make_figures.py [--out DIR] (writes fig3_forest.pdf,
fig4_bound.pdf, fig_checks.pdf, and figure_values.json to DIR, default this folder;
fig1_overview.pdf and fig2_timing.pdf are drawn diagrams and are not generated).
"""
import argparse
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.environ.get("AGENT_RISK_LOGS", os.path.abspath(os.path.join(HERE, "..", "..", "data", "logs")))
_parser = argparse.ArgumentParser(description="Figures D.1, D.2, and D.3 from the released analysis outputs.")
_parser.add_argument("--out", default=HERE, help="output folder (default: this folder)")
OUT = _parser.parse_args().out
os.makedirs(OUT, exist_ok=True)
GRAY, MINT, CORAL = "#C9C9C9", "#BFDFD2", "#ED8D5A"
MINT_DARK, CORAL_DARK, GRAY_DARK = "#4E9A7E", "#C0602C", "#8A8A8A"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "pdf.fonttype": 42,
})


def cp(k, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return float(lo), float(hi)


# ------------------------------------------------------------------ Figure 3
d58 = json.load(open(os.path.join(LOGS, "d58/analysis_d58.json")))
d64 = json.load(open(os.path.join(LOGS, "d64/analysis_d64.json")))
e7a = json.load(open(os.path.join(LOGS, "e7/analysis_e7.json")))
e7p = json.load(open(os.path.join(LOGS, "e7/prediction_e7.json")))
assert e7a["prediction"]["sha256"] == "a3bccf79de31846adfb037155d4f03017fad537fafabeacad3681fc6086b1bd7"
with open(os.path.join(LOGS, "e7/prediction_e7.json"), "rb") as f:
    assert hashlib.sha256(f.read()).hexdigest() == e7a["prediction"]["sha256"]
d66 = json.load(open(os.path.join(LOGS, "d66/analysis_d66.json")))
e2a = json.load(open(os.path.join(LOGS, "e2/analysis_e2.json")))
e3r = json.load(open(os.path.join(LOGS, "e3/report_e3.json")))
e2prim = e2a["primary_live_minus_replay_p_harm"]
e3prim = e3r["primary"]
e7g = {g: e7a["gates"][g] for g in ("1", "2")}
EQUIV = {}
e7skip = json.load(open(os.path.join(LOGS, "e7/posthoc_skip_reference_e7.json")))
assert e7skip["note_sha256"] == "8e7f1bbf6dbf8fafc77ba43b28cb1564b66ef46573b05ba6d4f208f3f1cdf964"
with open(os.path.join(LOGS, "e7/posthoc_skip_reference_e7.json"), "rb") as f:
    assert hashlib.sha256(f.read()).hexdigest() == "fb545757453cdea06cadd319bb9ff1d55c4ef5d86322ee49751aafdeb3d90f78"
FS = 8.0
ACC = {"match": MINT_DARK, "compatible": GRAY_DARK, "fails": CORAL_DARK}
FACE = {"match": MINT, "compatible": GRAY, "fails": CORAL}


def row(label, d, verdict, margin, key90="newcombe_paired_90", key95="newcombe_paired_95", equivalent=None, cluster90=None):
    if equivalent is not None:
        EQUIV[label] = equivalent
    pt = sum(d["point_range"]) / 2
    lo, hi = d[key95]
    inner = d[key90]
    excl = lo > 0 or hi < 0
    assert lo <= inner[0] <= inner[1] <= hi, label
    assert lo <= pt <= hi, label
    if verdict == "fails":
        assert excl and not EQUIV.get(label, False), label
    elif verdict == "compatible":
        assert not excl, label
    else:
        assert margin is not None and EQUIV.get(label, not excl), label
        assert -margin < inner[0] <= inner[1] < margin, label
        if cluster90 is not None:
            assert -margin < cluster90[0] <= cluster90[1] < margin, label
    return dict(label=label, point=pt, interval90=inner, interval95=[lo, hi], verdict=verdict,
                margin=margin, cluster90=cluster90)


def e7row(label, d, verdict):
    return row(label, d["newcombe"], verdict, d["delta"], equivalent=d["equivalent"],
               cluster90=d["cluster"]["cluster_boot_90"])


for g in ("1", "2"):
    sk = e7skip["gates"][g]
    assert sk["live"] == e7g[g]["live"]["p_harm"]
    assert sk["skip"]["n"] == e7g[g]["identified"]["p_harm"]["n"]
    assert sk["skip"]["range"] == [sk["skip"]["k"] / sk["skip"]["n"],
                                  (sk["skip"]["k"] + sk["skip"]["unknown"]) / sk["skip"]["n"]]
    assert not sk["contrast"]["equivalent"] and sk["contrast"]["newcombe95_excludes_0"]
    assert sk["contrast"]["newcombe"]["newcombe_paired_95"][1] < 0

p1 = d66["primary"]["P1_relay_L2_minus_second_clean_L2"]
p2 = d66["primary"]["P2_relay_L2_minus_second_snap_L2"]
d64u = d64["secondary_iii_unannounced_minus_baseline_replay_abort"]
d64a = d64["secondary_ii_announced_minus_baseline_replay_abort"]
rows = [
    e7row("Retry gate 1, targeted execution", e7g["1"]["primary"], "match"),
    e7row("Retry gate 2, targeted execution", e7g["2"]["primary"], "match"),
    e7row("Gate 1, retry counted as stop", e7g["1"]["references"]["retry_as_stop"]["contrast"], "fails"),
    e7row("Gate 1, skip blocked (post hoc)", e7skip["gates"]["1"]["contrast"], "fails"),
    row("Second agent, snapshot start", p2, "match", 0.03, equivalent=p2["equivalent_at_margin_90"]),
    row("Second agent, clean start", p1, "fails", 0.03, equivalent=p1["equivalent_at_margin_90"]),
    row("Unannounced gate, abort rate", d64u, "compatible", None),
    row("Announced gate, abort rate", d64a, "fails", None),
]
d58prim = d58["primary_live_minus_replay_p_harm"]
checks = [
    row("Stopping gate, harm", d58prim, "match", d58prim["margin"], "newcombe90_envelope", "newcombe95", d58prim["equivalent_at_margin"]),
    row("Stronger agent, harm", e2prim, "match", e2prim["delta"], equivalent=e2prim["equivalent"]),
    row("Airline, harm", e3prim, "match", e3r["delta"], "newcombe90", "newcombe95", cluster90=e3prim["cluster90"]),
    row("Stopping gate, abort rate", d58["secondary_iii"]["abort_rate"], "compatible", None, "newcombe90_envelope", "newcombe95"),
]


def forest(data, filename, height, separators=()):
    width, left, axw, bottom, top = 3.03, 1.24, 1.16, 0.37, 0.04
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left / width, bottom / height, axw / width, (height-bottom-top) / height])
    n = len(data)
    ax.set(xlim=(-0.09, 0.135), ylim=(-0.5, n-0.5))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.axvline(0, color="black", lw=0.6, zorder=1)
    for i, d in enumerate(data):
        y, m, verdict = n-1-i, d["margin"], d["verdict"]
        if m is not None:
            ax.add_patch(Rectangle((-m,y-0.36),2*m,0.72,fc=GRAY,alpha=0.45,lw=0,zorder=0))
        ax.plot(d["interval95"], [y,y], color=ACC[verdict], lw=0.7, solid_capstyle="butt", zorder=2)
        ax.plot(d["interval90"], [y,y], color=ACC[verdict], lw=2.5, solid_capstyle="butt", zorder=3)
        ax.plot([d["point"]], [y], marker="o", ms=2.2,mfc=FACE[verdict],mec="black",mew=0.4,zorder=4)
        ax.text(1.04,y+.23,verdict,transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=FS,color=ACC[verdict])
        ax.text(1.04,y-.23,format(d["point"],"+.3f").replace("-", "−"),transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=FS)
    labels = [d["label"].replace(", ", ",\n", 1) for d in data]
    ax.set_yticks(range(n), labels[::-1], fontsize=FS)
    ax.tick_params(axis="y",pad=3)
    for index in separators:
        y = n-index-0.5
        ax.plot([-.09,.135],[y,y],color=GRAY,lw=.45,zorder=0)
    ax.set_xticks([-.06,0,.06,.12],["−0.06","0","0.06","0.12"],fontsize=FS)
    fig.text(.5, .025 / height, "probability difference: live minus prediction",
             ha="center", va="bottom", fontsize=FS)
    fig.savefig(os.path.join(OUT,filename))
    plt.close(fig)


forest(rows,"fig3_forest.pdf",2.65,(4,6))
forest(checks,"fig_checks.pdf",1.53)

# Figure 4: registered references and the post hoc skip reading share the log-only row.
CEIL = e7a["decision_example"]["harm_ceiling"]
assert CEIL == 0.25


def prop_env(d):
    k,u,n = d["k"],d["unknown"],d["n"]
    return sum(d["range"])/2, (cp(k,n)[0],cp(k+u,n)[1])


fig4 = {}
for g in ("1","2"):
    gd = e7g[g]
    lb = gd["references"]["log_alone_bounds"]["p_harm_bounds"]
    unk = gd["identified"]["p_harm"]["unknown"]
    n = gd["identified"]["p_harm"]["n"]
    bounds = [lb[0],lb[1]+unk/n]
    refs = {k: gd["references"][k]["reference_value"] for k in ("retry_as_stop","pass_all")}
    assert abs(refs["retry_as_stop"]-bounds[0]) < 0.001
    if g == "1":
        assert abs(refs["pass_all"]-bounds[1]) < 0.001
    else:
        assert bounds[0] < refs["pass_all"] < bounds[1]
    skip = sum(e7skip["gates"][g]["skip"]["range"])/2
    assert bounds[0] <= skip <= bounds[1]
    fig4[g] = dict(log_alone=bounds, identified=prop_env(gd["identified"]["p_harm"]),
                   live=prop_env(gd["live"]["p_harm"]),skip=skip,references=refs,n=n,
                   n_cont=e7p["gates"][g]["decomposition"]["continuations_valid"],
                   cont_share=e7p["gates"][g]["decomposition"]["continuing_mass"])

W4,H4 = 3.03,2.40
fig = plt.figure(figsize=(W4,H4))
axes4=[]
for gi,g in enumerate(("1","2")):
    v=fig4[g]
    bottom=1.43 if gi==0 else .35
    ax=fig.add_axes([.78/W4,bottom/H4,2.17/W4,.76/H4])
    axes4.append(ax)
    ax.set(xlim=(.055,.56),ylim=(-.6,3.6))
    for s in ("top","right","left"):
        ax.spines[s].set_visible(False)
    ax.axvline(CEIL,color="black",lw=.7,ls=(0,(3,2)),zorder=1)
    lo,hi=v["log_alone"]
    ax.plot([lo,hi],[2.3,2.3],color=GRAY,lw=4.2,solid_capstyle="butt",zorder=2)
    ax.plot([lo,hi],[2.3,2.3],"|",color=GRAY_DARK,ms=5,mew=.7,zorder=3)
    ax.annotate("retry as stop",xy=(lo,3.02),xytext=(-3 if g == "1" else 0,0),
                textcoords="offset points",ha="right" if g == "1" else "left",va="center",fontsize=FS)
    ax.text(hi,3.15,"ignore gate" if g == "1" else "upper\nbound",ha="left" if g == "1" else "center",va="center",fontsize=FS)
    if g == "2":
        ref = v["references"]["pass_all"]
        ax.plot(ref,2.3,marker="s",ms=3.5,mfc="white",mec=GRAY_DARK,mew=.7,zorder=4)
        ax.text(ref,3.15,"ignore\ngate",ha="center",va="center",fontsize=FS)
    ax.plot([lo,lo],[2.6,2.84],color=GRAY_DARK,lw=.5)
    ax.plot([hi,hi],[2.6,2.84],color=GRAY_DARK,lw=.5)
    ax.plot(v["skip"],2.3,marker="D",ms=4,mfc="white",mec=CORAL_DARK,mew=.8,zorder=4)
    ax.text(v["skip"],1.57,"skip blocked",ha="left" if g == "1" else "right",va="center",fontsize=FS,color=CORAL_DARK)
    for y,(pt,(a,b)),col,face in ((.75,v["identified"],MINT_DARK,MINT),(-.2,v["live"],"black","white")):
        ax.plot([a,b],[y,y],color=col,lw=1.6,zorder=2)
        ax.plot(pt,y,marker="o",ms=3.5,mfc=face,mec="black",mew=.5,zorder=3)
    ax.set_yticks([2.3,.75,-.2],["log alone","log + executed","live gate"],fontsize=FS)
    ax.tick_params(axis="y",pad=3)
    fig.text(.015,(bottom+.76+.07)/H4,f"Retry gate {g}",fontsize=FS,fontweight="bold",ha="left",va="bottom")
    if gi==0:
        ax.tick_params(axis="x",labelbottom=False,length=0)
        ax.spines["bottom"].set_visible(False)
axes4[-1].set_xticks([.1,.2,.3,.4],["0.1","0.2","0.3","0.4"],fontsize=FS)
axes4[-1].set_xlabel(f"P(harm); dashed: ceiling {CEIL:.2f}",fontsize=FS,labelpad=2.5)
fig.savefig(os.path.join(OUT,"fig4_bound.pdf"))
plt.close(fig)

# arm E agreement table (bi_baseline, full interface): (label, mass k, first-decision k)
ladder = [("same model", 99, 99), ("other kernel mode", 2, 7), ("precision-only update", 1, 3),
          ("release (pooled)", 0, 0)]
N = 99

# Appendix figure (model releases): moved out of Figure 4 in the page-fit pass of October 6.
fig = plt.figure(figsize=(3.25, 1.15))
axl = fig.add_subplot(1, 1, 1)
for i, (lab, km, kf) in enumerate(ladder):
    y = len(ladder) - 1 - i
    if km == N:
        axl.text(0.003, y, "99/99 (control; off axis)", fontsize=FS, va="center")
        continue
    mlo, mhi = cp(km, N)
    flo, fhi = cp(kf, N)
    axl.plot([mlo, mhi], [y + 0.14, y + 0.14], color=MINT_DARK, lw=1.4)
    axl.plot([km / N], [y + 0.14], marker="o", ms=3.8, mfc=MINT, mec="black", mew=0.5)
    axl.plot([flo, fhi], [y - 0.14, y - 0.14], color="black", lw=0.8)
    axl.plot([kf / N], [y - 0.14], marker="s", ms=3.2, mfc="white", mec="black", mew=0.6)
axl.set_yticks(range(len(ladder)))
axl.set_yticklabels([l[0] for l in ladder][::-1], fontsize=FS)
axl.set_xlim(-0.004, 0.15)
axl.set_ylim(-0.5, len(ladder) - 0.5)
axl.set_xticks([0, 0.05, 0.10, 0.15])
axl.set_xticklabels(["0", "0.05", "0.10", "0.15"], fontsize=FS)
axl.tick_params(axis="y", length=0)
axl.set_xlabel("share of 99 logged episodes", fontsize=FS, labelpad=1.5)
axl.plot([], [], marker="o", ms=3.8, mfc=MINT, mec="black", mew=0.5, color=MINT_DARK, lw=1.4, label="agreement mass")
axl.plot([], [], marker="s", ms=3.2, mfc="white", mec="black", mew=0.6, color="black", lw=0.8, label="first decision")
axl.legend(loc="upper right", fontsize=FS, frameon=False, handlelength=1.6, borderaxespad=0.0, labelspacing=0.25)
for s in ("top", "right"):
    axl.spines[s].set_visible(False)
fig.subplots_adjust(left=0.32, right=0.97, top=0.97, bottom=0.29)
# The existing release figure is outside this revision.
plt.close(fig)

summary = {
    "fig3_rows": rows,
    "fig_checks_rows": checks,
    "fig4": fig4,
    "fig4b": [(lab, km, cp(km, N), kf, cp(kf, N)) for lab, km, kf in ladder],
    # Retain the frozen manifest's original layout metadata for byte identity.
    "heights_inches": {"fig3_forest": 1.85, "fig_checks": 1.08, "fig4_bound": 1.57},
}
json.dump(summary, open(os.path.join(OUT, "figure_values.json"), "w"), indent=1)
print("Figure assertions passed; wrote Figure 3, Figure 4, and appendix checks.")
