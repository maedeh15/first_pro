"""
xai_evidence_map.py -- one-page summary figure of the XAI chapter: model part x research question, colour = verdict.
The verdicts are WRITTEN HERE by hand from the chapter's claims ledger (thesis_chapter_xai_draft.md); edit CELLS if a result changes.
  python xai_evidence_map.py --out-dir figs        (writes fig_evidence_map.png/.pdf)
"""
import argparse, os
import fig_style as fs

# verdict codes: ok = supported / validated, part = partial / weak / mixed, no = rejected / invalid, na = not tested
COL = {"ok": "#009E73", "part": "#E69F00", "no": "#D55E00", "na": "#BBBBBB"}
QS = ["Q1  Is the explanation trustworthy?", "Q2  What does the model use?", "Q3  Do literature claims hold?"]
ROWS = ["fNIRS branch, segment level\n(Grad-CAM, occlusion)", "fNIRS branch, gradient methods\nand GradientShap", "fNIRS head, linear variant\n(exact segment evidence, seed 999)", "EGTA attention", "CAFE gates", "EFGF scalars (gamma, alpha)",
        "Deep EEG branch", "Linear Riemannian member", "Time axis (segment profile)"]
CELLS = [
    [("ok", "passes V1 + V2\nrho 0.30 with occlusion"), ("ok", "both chromophores;\ncontent < 0.1 Hz"), ("no", "HbO rises with load: reversed;\ninverted U: not supported")],
    [("part", "weak (rho 0.04-0.09);\ncompleteness violated"), ("na", "not analysed"), ("na", "-")],
    [("ok", "exact additivity; rho 0.49 with\nocclusion; top-3 deletion 26/26"), ("part", "late segments weigh more;\nfusion path matters in 7 subjects"), ("na", "stability over seeds\nnot tested")],
    [("no", "near-uniform\n(entropy 0.983)"), ("na", "-"), ("na", "-")],
    [("no", "stay near 0.5"), ("na", "-"), ("na", "-")],
    [("na", "not tested\n(script ready)"), ("na", "-"), ("na", "-")],
    [("na", "not explained directly\n(head accuracy ~0.48)"), ("ok", "alpha band used"), ("part", "alpha down: yes (global);\nfrontal theta: 0v2 only")],
    [("part", "exact SHAP; channel ranking\ncausal only for coefficients"), ("ok", "inter-channel covariance,\nalpha / theta"), ("part", "SHAP: frontal theta yes,\nparietal alpha no (sign ambiguity)")],
    [("part", "exploratory;\nsegment-4 spike (0-back)"), ("part", "late segments weigh more;\nnot class-specific"), ("na", "-")],
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="figs"); a = ap.parse_args(); fs.apply()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    fig, ax = plt.subplots(figsize=(fs.W2 * 1.35, 5.1)); nr, nc = len(ROWS), len(QS)
    for i, row in enumerate(CELLS):
        for j, (v, txt) in enumerate(row):
            ax.add_patch(Rectangle((j, nr - 1 - i), 0.96, 0.92, facecolor=COL[v], alpha=0.88 if v != "na" else 0.45, edgecolor="white"))
            ax.text(j + 0.48, nr - 1 - i + 0.46, txt, ha="center", va="center", fontsize=6.3, color="black" if v in ("na", "part") else "white")
    ax.set_xlim(0, nc); ax.set_ylim(0, nr); ax.set_xticks([j + 0.48 for j in range(nc)]); ax.set_xticklabels(QS, fontsize=7); ax.xaxis.tick_top()
    ax.set_yticks([nr - 1 - i + 0.46 for i in range(nr)]); ax.set_yticklabels(ROWS, fontsize=6.8); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.legend(handles=[Patch(color=COL["ok"], label="supported / validated"), Patch(color=COL["part"], label="partial / weak / mixed"), Patch(color=COL["no"], label="rejected / invalid"), Patch(color=COL["na"], alpha=0.45, label="not tested")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=6.5)
    ax.set_title("Evidence map of the explainability analysis (26 subjects; verdicts as in the claims ledger)", fontsize=8, pad=22)
    fig.tight_layout(); print("figure:", fs.save(fig, a.out_dir, "fig_evidence_map"))


if __name__ == "__main__":
    main()
