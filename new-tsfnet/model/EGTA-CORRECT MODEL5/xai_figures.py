"""
xai_figures.py -- thesis-ready figures (PNG 300 dpi + vector PDF) from the EXISTING XAI outputs; CPU only, no model needed.
Every figure shows n and the unit (subject), one dot per subject, mean and 95% bootstrap CI; Okabe-Ito palette (fig_style.py). A panel is skipped
(with a message) when its input file is absent, so you can run it on whatever you have.

  python xai_figures.py --out-dir figs --validated validated_maps_summary.json --profiles segment_profiles_by_class.json \
      --gradient gradient_faithfulness_per_subject.csv --consensus xai_consensus_per_subject.csv --stability xai_stability_per_subject.csv \
      --faithfulness faithfulness_per_subject.csv --psd fnirs_psd_per_subject.csv --folds ensemble_folds.csv --baseline baseline_all_subjects.csv
Figures: fig_validated_maps (V1 randomisation rho, V2 deletion top-random, V3 impact rho) | fig_segment_profiles | fig_gradient_faithfulness |
fig_consensus | fig_stability | fig_attention_faithfulness | fig_head_heterogeneity (claim 9)
"""
import argparse, json, os, sys
import numpy as np
import fig_style as fs

OK = lambda p: p and os.path.isfile(p)


def _rows(ax, names, data, colors, horizontal=True, xlabel=""):
    """one row per name: per-subject dots, mean, 95% bootstrap CI"""
    for i, (nm, v) in enumerate(zip(names, data)):
        v = np.asarray(v, float); v = v[~np.isnan(v)]
        fs.strip(ax, i, v, color=colors[i], horizontal=horizontal); fs.mean_ci(ax, i, v, color="k", horizontal=horizontal)
    (ax.set_yticks if horizontal else ax.set_xticks)(range(len(names))); (ax.set_yticklabels if horizontal else ax.set_xticklabels)(names)
    if horizontal: ax.invert_yaxis(); ax.set_xlabel(xlabel)
    else: ax.set_ylabel(xlabel)


def fig_validated(path, out):
    import matplotlib.pyplot as plt
    J = json.load(open(path))["per_subject"]; ids = list(J); ms = [m for m in fs.METHOD_LABEL if all(m in J[s] for s in ids)]
    labs = [fs.METHOD_LABEL[m] for m in ms]; cols = [fs.METHOD_COLOR[m] for m in ms]
    fig, ax = plt.subplots(1, 3, figsize=(fs.W2 * 1.15, 2.9), sharey=False)
    _rows(ax[0], labs, [[J[s][m]["rand_rho"] for s in ids] for m in ms], cols, xlabel="rho: trained vs\nrandomised model"); ax[0].axvline(0.3, ls="--", color="k", lw=0.6)
    _rows(ax[1], labs, [[J[s][m]["top"] - J[s][m]["random"] for s in ids] for m in ms], cols, xlabel="Deletion: top minus random\ndrop in P(class)"); ax[1].axvline(0, color="k", lw=0.6)
    ms3 = [m for m in ms if m != "occlusion"]
    _rows(ax[2], [fs.METHOD_LABEL[m] for m in ms3], [[J[s][m]["impact_rho"] for s in ids] for m in ms3], [fs.METHOD_COLOR[m] for m in ms3], xlabel="rho vs occlusion\nimpact"); ax[2].axvline(0, color="k", lw=0.6)
    for a in ax[1:]: a.set_yticklabels([]) if False else None
    for a, l in zip(ax, "abc"): fs.panel(a, l)
    fig.suptitle(f"Validation of explanation methods (n = {len(ids)} subjects, fNIRS segment level)", fontsize=8, y=1.0); fig.tight_layout(); return fs.save(fig, out, "fig_validated_maps")


def fig_profiles(path, out):
    import matplotlib.pyplot as plt
    P = json.load(open(path)); ids = list(P); ms = [m for m in ("occlusion", "grad_cam", "input_gradient") if all(m in P[s] for s in ids)]
    names = {"0": "0-back", "1": "2-back", "2": "3-back"}; cc = {"0": fs.C["blue"], "1": fs.C["orange"], "2": fs.C["red"]}
    fig, ax = plt.subplots(1, len(ms), figsize=(fs.W2, 2.6), sharey=False)
    ax = np.atleast_1d(ax)
    for a, m in zip(ax, ms):
        for cl in names:
            A = np.array([P[s][m][cl] for s in ids if cl in P[s][m]]); x = np.arange(A.shape[1])
            mu = A.mean(0); lo, hi = np.array([fs.boot_ci(A[:, k], n=1000) for k in x]).T
            a.fill_between(x + 1, lo, hi, color=cc[cl], alpha=0.2, lw=0); a.plot(x + 1, mu, color=cc[cl], label=names[cl])
        a.axhline(0, color="k", lw=0.5); a.set_xlabel("Segment position in window (1 s steps)"); a.set_title(fs.METHOD_LABEL[m], fontsize=8)
    ax[0].set_ylabel("Importance (within-window z)"); ax[0].legend(frameon=False, fontsize=6)
    for a, l in zip(ax, "abc"): fs.panel(a, l)
    fig.tight_layout(); return fs.save(fig, out, "fig_segment_profiles")


def fig_gradient(path, out):
    import pandas as pd, matplotlib.pyplot as plt
    d = pd.read_csv(path); fig, ax = plt.subplots(1, 2, figsize=(fs.W2, 2.5))
    _rows(ax[0], ["Input gradient", "Gradient x input"], [d["rho_ig"], d["rho_gxi"]], [fs.C["orange"], fs.C["green"]], xlabel="Spearman rho vs occlusion impact"); ax[0].axvline(0, color="k", lw=0.6)
    cols = [c for c in d.columns if c.startswith("drop_")]; order = [c for c in ("drop_top", "drop_random", "drop_bottom") if c in cols]
    _rows(ax[1], [c[5:].capitalize() for c in order], [d[c] for c in order], [fs.C["red"], fs.C["grey"], fs.C["blue"]][:len(order)], xlabel="Drop in P(class) after deleting 3 segments")
    for a, l in zip(ax, "ab"): fs.panel(a, l)
    fig.suptitle(f"Gradient faithfulness (n = {len(d)} subjects)", fontsize=8); fig.tight_layout(); return fs.save(fig, out, "fig_gradient_faithfulness")


def fig_consensus(path, out):
    import pandas as pd, matplotlib.pyplot as plt
    d = pd.read_csv(path); cols = [c for c in d.columns if c.endswith("_mean_rho")]
    KNOWN = {"attention": "EGTA attention", "input_gradient": "Input gradient", "integrated_gradients": "Integrated gradients", "grad_x_input": "Gradient x input", "grad_cam": "Grad-CAM", "gradient_shap": "GradientShap", "occlusion": "Occlusion"}
    def pair_label(c):
        core = c[:-len("_mean_rho")]
        for a in sorted(KNOWN, key=len, reverse=True):
            if core.startswith(a + "_") and core[len(a) + 1:] in KNOWN: return f"{KNOWN[a]} vs {KNOWN[core[len(a) + 1:]].lower() if False else KNOWN[core[len(a) + 1:]]}"
        return core.replace("_", " ")
    names = [pair_label(c) for c in cols]; fig, ax = plt.subplots(figsize=(fs.W1 * 1.7, 2.2))
    _rows(ax, names, [d[c] for c in cols], [fs.C["purple"]] * len(cols), xlabel="Per-subject mean Spearman rho between methods"); ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"Agreement between explanation methods (n = {len(d)})", fontsize=8); fig.tight_layout(); return fs.save(fig, out, "fig_consensus")


def fig_stability(path, folds, out):
    import pandas as pd, matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    d = pd.read_csv(path).sort_values("icc_2_1"); fig, ax = plt.subplots(1, 3 if OK(folds) else 2, figsize=(fs.W2, 2.6))
    col = [fs.C["red"] if v < 0.5 else fs.C["orange"] if v < 0.75 else fs.C["sky"] if v < 0.9 else fs.C["green"] for v in d["icc_2_1"]]
    ax[0].bar(range(len(d)), d["icc_2_1"], color=col); ax[0].set_xticks(range(len(d))); ax[0].set_xticklabels([s[2:] for s in d["subject"]], rotation=90, fontsize=5.5)
    for y in (0.5, 0.75, 0.9): ax[0].axhline(y, ls=":", color="k", lw=0.5)
    ax[0].set_ylabel("ICC(2,1) of attention across seeds"); ax[0].set_xlabel("Subject")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=fs.C["red"], label="poor < 0.5"), Patch(color=fs.C["orange"], label="moderate 0.5-0.75"), Patch(color=fs.C["sky"], label="good 0.75-0.9"), Patch(color=fs.C["green"], label="excellent > 0.9")], frameon=False, fontsize=6, loc="upper left", title="Koo and Li", title_fontsize=6)
    ax[1].scatter(d["icc_2_1"], d["mean_pairwise_rho"], s=14, color=fs.C["blue"]); ax[1].set_xlabel("ICC(2,1)"); ax[1].set_ylabel("Mean pairwise rho")
    if OK(folds):
        fo = pd.read_csv(folds); sc = next(c for c in fo.columns if c.lower() in ("subject", "subject_id", "sid"))
        ac = next((c for c in fo.columns if "ens" in c.lower() and "win" in c.lower() and "acc" in c.lower()), next(c for c in fo.columns if "acc" in c.lower()))
        m = d.merge(fo.groupby(sc)[ac].mean().rename("acc"), left_on="subject", right_index=True); r, p = spearmanr(m["icc_2_1"], m["acc"])
        ax[2].scatter(m["icc_2_1"], m["acc"] * (100 if m["acc"].max() <= 1 else 1), s=14, color=fs.C["blue"]); ax[2].set_xlabel("ICC(2,1)"); ax[2].set_ylabel("Subject accuracy (%)")
        ax[2].set_title(f"Spearman {r:+.2f}, p = {p:.2g} (exploratory)", fontsize=7)
    for a, l in zip(ax, "abc"): fs.panel(a, l)
    fig.tight_layout(); return fs.save(fig, out, "fig_stability")


def fig_attn_faith(path, out):
    import pandas as pd, matplotlib.pyplot as plt
    d = pd.read_csv(path); fig, ax = plt.subplots(figsize=(fs.W1 * 1.5, 2.0))
    _rows(ax, ["EGTA", "EFGF1"], [d["egta_mean_rho"], d["efgf1_mean_rho"]], [fs.C["blue"], fs.C["orange"]], xlabel="Spearman rho, attention vs occlusion impact"); ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"Attention faithfulness (n = {len(d)})", fontsize=8); fig.tight_layout(); return fs.save(fig, out, "fig_attention_faithfulness")


def fig_heterogeneity(path, out):
    """claim 9: per-subject accuracy of the EEG, fNIRS and fusion heads (paired lines) and the dominant head"""
    import pandas as pd, matplotlib.pyplot as plt
    d = pd.read_csv(path); d = d[d.get("status", "ok").astype(str).str.lower().isin(["ok", "done", "success"])] if "status" in d else d
    cols = [c for c in ("eeg_only_acc", "fusion_only_acc", "fnirs_only_acc") if c in d]; names = {"eeg_only_acc": "EEG head", "fusion_only_acc": "Fusion head", "fnirs_only_acc": "fNIRS head"}
    colr = {"eeg_only_acc": fs.C["blue"], "fusion_only_acc": fs.C["purple"], "fnirs_only_acc": fs.C["red"]}
    if len(cols) < 2: print("[skip] heterogeneity: head columns missing"); return None
    v = d[cols].values * (100 if d[cols].values.max() <= 1 else 1); fig, ax = plt.subplots(1, 2, figsize=(fs.W2, 2.8))
    for row in v: ax[0].plot(range(len(cols)), row, color="0.75", lw=0.6, zorder=1)
    for i, c in enumerate(cols): fs.strip(ax[0], i, v[:, i], color=colr[c]); fs.mean_ci(ax[0], i, v[:, i], color="k")
    ax[0].set_xticks(range(len(cols))); ax[0].set_xticklabels([names[c] for c in cols]); ax[0].set_ylabel("Window accuracy (%)"); ax[0].axhline(100 / 3, ls="--", color="k", lw=0.6)
    dom = np.array(cols)[v.argmax(1)]; cnt = [int((dom == c).sum()) for c in cols]
    ax[1].bar(range(len(cols)), cnt, color=[colr[c] for c in cols]); ax[1].set_xticks(range(len(cols))); ax[1].set_xticklabels([names[c] for c in cols]); ax[1].set_ylabel("Subjects for whom this head is best")
    ax[0].set_title(f"n = {len(d)} subjects", fontsize=8)
    for a, l in zip(ax, "ab"): fs.panel(a, l)
    fig.tight_layout(); return fs.save(fig, out, "fig_head_heterogeneity")


def self_test():
    import tempfile, pandas as pd
    fs.apply(); rng = np.random.RandomState(0); d = tempfile.mkdtemp(); ids = [f"VP{i:03d}" for i in range(1, 15)]; ms = list(fs.METHOD_LABEL)[:6]
    J = {s: {m: {"rand_rho": rng.rand() * .3, "impact_rho": rng.rand(), "top": rng.rand() * .1, "random": rng.rand() * .05, "bottom": rng.rand() * .02} for m in ms} for s in ids}
    json.dump({"per_subject": J}, open(f"{d}/v.json", "w")); P = {s: {m: {str(c): list(rng.randn(21)) for c in range(3)} for m in ms} for s in ids}; json.dump(P, open(f"{d}/p.json", "w"))
    pd.DataFrame({"subject": ids, "rho_ig": rng.rand(14), "rho_gxi": rng.rand(14), "n": 50, "drop_top": rng.rand(14), "drop_random": rng.rand(14) * .5, "drop_bottom": rng.rand(14) * .2}).to_csv(f"{d}/g.csv", index=False)
    pd.DataFrame({"subject": ids, **{f"{a}_{b}_mean_rho": rng.rand(14) for a, b in (("attention", "input_gradient"), ("attention", "integrated_gradients"))}}).to_csv(f"{d}/c.csv", index=False)
    pd.DataFrame({"subject": ids, "n_windows": 50, "mean_pairwise_rho": rng.rand(14), "sd_pairwise_rho": .1, "icc_2_1": rng.rand(14)}).to_csv(f"{d}/s.csv", index=False)
    pd.DataFrame({"subject_id": ids * 3, "mean_acc": rng.rand(42)}).to_csv(f"{d}/f.csv", index=False)
    pd.DataFrame({"subject": ids, "egta_mean_rho": rng.rand(14), "efgf1_mean_rho": rng.rand(14)}).to_csv(f"{d}/a.csv", index=False)
    pd.DataFrame({"subject_id": ids, "status": "ok", "eeg_only_acc": rng.rand(14), "fusion_only_acc": rng.rand(14), "fnirs_only_acc": rng.rand(14)}).to_csv(f"{d}/b.csv", index=False)
    outs = [fig_validated(f"{d}/v.json", d), fig_profiles(f"{d}/p.json", d), fig_gradient(f"{d}/g.csv", d), fig_consensus(f"{d}/c.csv", d), fig_stability(f"{d}/s.csv", f"{d}/f.csv", d),
            fig_attn_faith(f"{d}/a.csv", d), fig_heterogeneity(f"{d}/b.csv", d)]
    assert all(o and os.path.isfile(o) and os.path.isfile(o[:-4] + ".pdf") for o in outs); print(f"[1] {len(outs)} figures written (PNG + PDF)\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("validated", "profiles", "gradient", "consensus", "stability", "faithfulness", "folds", "baseline"): ap.add_argument("--" + k)
    ap.add_argument("--out-dir", default="figs"); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    fs.apply(); os.makedirs(a.out_dir, exist_ok=True)
    jobs = [("validated", lambda: fig_validated(a.validated, a.out_dir)), ("profiles", lambda: fig_profiles(a.profiles, a.out_dir)), ("gradient", lambda: fig_gradient(a.gradient, a.out_dir)),
            ("consensus", lambda: fig_consensus(a.consensus, a.out_dir)), ("stability", lambda: fig_stability(a.stability, a.folds, a.out_dir)),
            ("faithfulness", lambda: fig_attn_faith(a.faithfulness, a.out_dir)), ("baseline", lambda: fig_heterogeneity(a.baseline, a.out_dir))]
    for k, f in jobs:
        if not OK(getattr(a, k)): print(f"[skip] --{k} not given / not found"); continue
        try: print("figure:", f())
        except Exception as e: print(f"[{k}] FAILED -- {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
