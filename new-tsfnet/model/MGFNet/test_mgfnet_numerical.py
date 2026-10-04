"""
test_mgfnet_numerical.py -- MANDATORY pre-GPU check for the MGFNet code.
CPU, synthetic data, the REAL classes. Run:  python test_mgfnet_numerical.py
"""
import sys

import numpy as np
import torch
import torch.nn as nn

from mgfnet_model import (MGFNet, mgfnet_deep_supervision_loss, EEG_F_RANGE,
                          NIRS_F_RANGE)
from mgfnet_dataset import ChannelNormalizer, stratified_trial_kfold
from mgfnet_train import itr_bits_per_min, macro_prf, select_epoch_cv

B, T_EEG, T_NIRS = 4, 1000, 50           # the paper's 5 s window


def batch(c_eeg=30, t_eeg=T_EEG, t_nirs=T_NIRS, b=B):
    return (torch.randn(b, c_eeg, t_eeg), torch.randn(b, 36, t_nirs), torch.randn(b, 36, t_nirs))


def report(name, ok, detail=""):
    print(f"[{name}] {detail}  {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def check_paper_dims():
    d30 = MGFNet(eeg_channels=30).branch_dims
    d28 = MGFNet(eeg_channels=28).branch_dims
    exp30 = {"1": 120, "2": 144, "3": 144, "12": 264, "13": 264, "23": 288}
    exp28 = {"1": 112, "2": 144, "3": 144, "12": 256, "13": 256, "23": 288}
    return report("paper_dims", d30 == exp30 and d28 == exp28,
                  f"30ch={d30} | 28ch={d28}")


def check_shapes():
    ok = True
    for c in (30, 28):
        m = MGFNet(eeg_channels=c).eval()
        for te, tn in ((1000, 50), (2000, 100)):
            with torch.no_grad():
                y, ya, aux = m(*batch(c, te, tn))
            ok &= (y.shape == (B, 3) and len(ya) == 6 and aux["gate_weights"].shape == (B, 6)
                   and aux["coupling"].shape == (B, 5, 3)
                   and aux["component_selection"].shape == (B, 15))
    return report("shapes", ok, "30/28 EEG channels x 5 s/10 s windows")


def _grads(m, c=30):
    torch.manual_seed(0)
    m.train()
    y, ya, _ = m(*batch(c, b=16))
    mgfnet_deep_supervision_loss(y, ya, torch.randint(0, 3, (16,))).backward()


def check_gradients_everywhere():
    m = MGFNet()
    _grads(m)
    none_g = [n for n, p in m.named_parameters() if p.grad is None]
    zero_g = [n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().max() == 0]
    bad_g = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    return report("grads_everywhere", not (none_g or zero_g or bad_g),
                  f"None:{none_g or '-'} zero:{zero_g or '-'} non-finite:{bad_g or '-'}")


def check_cgscr_critical_path():
    """RELATIVE threshold: every CGSCR-critical tensor's max |grad| must be at
    least 1e-3 x the median max |grad| over all CGSCR parameters. The original
    '> 0' test passed q/k gradients of 1e-8 (coupling erased by LayerNorm)."""
    m = MGFNet()
    _grads(m)
    c = m.cgscr
    gmax = {n: p.grad.abs().max().item() for n, p in c.named_parameters()}
    med = float(np.median(list(gmax.values())))
    targets = {"morlet f0 (EEG)": c.eeg_bank.f0_raw, "morlet bw (EEG)": c.eeg_bank.bw_raw,
               "morlet f0 (HbO)": c.hbo_bank.f0_raw, "coupling Q": c.q_proj.weight,
               "coupling K": c.k_proj.weight, "gumbel log_tau": c.log_tau,
               "routing gate": c.routing_gate[0].weight}
    ok = True
    for name, p in targets.items():
        g = p.grad.abs().max().item()
        good = np.isfinite(g) and g > 1e-3 * med
        ok &= good
        print(f"    {name:<16} |grad|max={g:.3e}  ratio to median={g / med:.2e}  "
              f"{'ok' if good else 'TOO SMALL'}")
    return report("cgscr_critical_path", ok, f"median CGSCR |grad|max={med:.3e}")


def check_coupling_changes_routing():
    """Behavioural: forcing two different coupling matrices must change the
    routing feature. (Before the fix, LN(C*x) == LN(x) made this ~0.)"""
    torch.manual_seed(0)
    m = MGFNet().eval()
    c = m.cgscr
    xe, xo, xr = batch(b=8)
    with torch.no_grad():
        Ep = c.proj_e(c.eeg_enc(c.eeg_bank(xe)))
        Fp = c.proj_f(0.5 * (c.nirs_enc(c.hbo_bank(xo)) + c.nirs_enc(c.hbr_bank(xr))))
        U = torch.full((8, 5, 3), 1 / 3)
        O = torch.zeros(8, 5, 3)
        O[..., 0] = 1.0
        a, _, sa = c.route(Ep, Fp, U)
        b, _, sb = c.route(Ep, Fp, O)
    rel = ((a - b).abs().mean() / a.abs().mean()).item()
    srel = ((sa - sb).abs().mean() / sa.abs().mean()).item()
    return report("coupling_changes_routing", rel > 0.05 and srel > 0.05,
                  f"relative change: x_scr={rel:.3f} scores={srel:.3f}")


def check_routing_discrete_and_deterministic():
    torch.manual_seed(0)
    m = MGFNet()
    xs = batch(b=8)
    m.train()
    _, _, a_tr = m(*xs)
    m.eval()
    with torch.no_grad():
        _, _, a1 = m(*xs)
        _, _, a2 = m(*xs)
    onehot = lambda s: bool(((s == 0) | (s == 1)).all() and (s.sum(-1) == 1).all())
    ok = (onehot(a_tr["component_selection"].detach()) and onehot(a1["component_selection"])
          and torch.equal(a1["component_selection"], a2["component_selection"]))
    return report("routing_discrete", ok, "one-hot in train AND eval; eval deterministic")


def check_morlet_init_in_band():
    c = MGFNet().cgscr
    fe = c.eeg_bank.f0.detach()
    fn = c.hbo_bank.f0.detach()
    ok = bool((fe > EEG_F_RANGE[0]).all() and (fe < EEG_F_RANGE[1]).all()
              and (fn > NIRS_F_RANGE[0]).all() and (fn < 0.2).all())
    return report("morlet_init", ok, f"EEG f0={fe.numpy().round(2)} fNIRS f0={fn.numpy().round(3)}")


def check_logits_and_simplices():
    torch.manual_seed(0)
    m = MGFNet().eval()
    with torch.no_grad():
        y, ya, aux = m(*batch())
    is_prob = bool(((y.sum(-1) - 1).abs() < 1e-4).all() and (y >= 0).all())
    aux_prob = any(bool(((v.sum(-1) - 1).abs() < 1e-4).all() and (v >= 0).all()) for v in ya.values())
    gate_ok = bool((aux["gate_weights"].sum(-1) - 1).abs().max() < 1e-4)
    coup_ok = bool((aux["coupling"].sum(-1) - 1).abs().max() < 1e-4)
    return report("logits", not is_prob and not aux_prob and gate_ok and coup_ok,
                  f"heads are logits; gate/coupling rows sum to 1")


def check_gate_per_sample():
    torch.manual_seed(0)
    m = MGFNet().eval()
    xs = batch(b=32)
    nn.init.normal_(m.cgscr.routing_gate[3].weight, std=1.0)
    nn.init.normal_(m.cgscr.routing_gate[3].bias, std=1.0)
    with torch.no_grad():
        _, _, aux = m(*xs)
    g = aux["gate_weights"]
    cv = (g.std(dim=0) / (g.mean(dim=0) + 1e-12)).max().item()
    return report("gate_per_sample", cv > 0.01, f"max CV={cv:.3e} (structural capability)")


def check_ablations_and_guards():
    ok = True
    for kw in (dict(use_cgscr=False), dict(use_cross_modal=False), dict(use_intra_modal=False)):
        m = MGFNet(**kw).eval()
        with torch.no_grad():
            y, ya, _ = m(*batch())
        ok &= y.shape == (B, 3) and len(ya) == len(m.active_keys)
    try:
        MGFNet(use_intra_modal=False, use_cross_modal=False)
        ok = False
    except ValueError:
        pass
    try:
        MGFNet(eeg_channels=30)(*batch(28))
        ok = False
    except ValueError:
        pass
    return report("ablations_guards", ok, "Table-4 variants build; both-off and channel-mismatch raise")


def check_overfit_tiny_batch():
    torch.manual_seed(0)
    m = MGFNet().train()
    xs = batch(t_eeg=400, t_nirs=20, b=8)
    y_true = torch.randint(0, 3, (8,))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    first = last = None
    for step in range(60):
        opt.zero_grad()
        y, ya, _ = m(*xs)
        loss = mgfnet_deep_supervision_loss(y, ya, y_true)
        loss.backward()
        opt.step()
        first = loss.item() if step == 0 else first
        last = loss.item()
    return report("overfit_tiny_batch", last < first * 0.5, f"loss {first:.4f} -> {last:.4f}")


def check_splits_and_normalizer():
    # 18 trials (2 sessions x 9), 36 windows each, 6 trials per class
    trial_ids = np.repeat(np.arange(18), 36)
    trial_lab = np.tile([0, 1, 2], 6)
    labels = trial_lab[trial_ids]
    folds = stratified_trial_kfold(trial_ids, labels, n_splits=5, seed=1)
    ok = len(folds) == 5
    covered = np.zeros(len(trial_ids), dtype=int)
    for tr, va in folds:
        ok &= not (set(trial_ids[tr]) & set(trial_ids[va]))          # trial-level
        ok &= set(labels[va]) == {0, 1, 2} or len(set(trial_ids[va])) < 3
        covered[va] += 1
    ok &= bool((covered == 1).all())                                  # each window validated once

    class _DS:
        pass
    ds = _DS()
    g = torch.Generator().manual_seed(0)
    ds.eeg, ds.hbo, ds.hbr = (torch.randn(20, 3, 10, generator=g) for _ in range(3))
    tr = np.arange(10)
    n1 = ChannelNormalizer.fit(ds, tr)
    ds.eeg[10:] += 1000.0                                             # change NON-train rows
    n2 = ChannelNormalizer.fit(ds, tr)
    ok &= all(torch.equal(n1.stats[k][0], n2.stats[k][0]) for k in n1.stats)
    xe, _, _ = n1(ds.eeg[tr], ds.hbo[tr], ds.hbr[tr])
    ok &= abs(xe.mean().item()) < 1e-5 and abs(xe.std(unbiased=False).item() - 1) < 1e-3
    return report("splits_normalizer", ok, "trial-level 5-fold, every window validated once; "
                                           "normaliser ignores non-train rows")


def check_metrics():
    # ITR must reproduce the paper's Tables 2-3 (T = 5 s)
    itr_nb = itr_bits_per_min(0.9940, 3, 5.0)
    itr_wg = itr_bits_per_min(0.9903, 2, 5.0)
    itr_mbc = itr_bits_per_min(0.9813, 3, 5.0)
    ok = abs(itr_nb - 18.31) < 0.01 and abs(itr_wg - 11.05) < 0.01 and abs(itr_mbc - 17.19) < 0.01
    p, r, f1 = macro_prf([[10, 0, 0], [0, 5, 5], [0, 0, 10]])
    ok &= abs(p - (1 + 1 + 10 / 15) / 3) < 1e-9 and abs(r - (1 + 0.5 + 1) / 3) < 1e-9
    best, _, _ = select_epoch_cv([[0.5, 0.8, 0.8], [0.5, 0.8, 0.8]],
                                 [[1.0, 0.6, 0.4], [1.0, 0.6, 0.4]])
    ok &= best == 3                                                   # tie -> lower loss
    return report("metrics", ok, f"ITR n-back={itr_nb:.2f} (paper 18.31), WG={itr_wg:.2f} "
                                 f"(paper 11.05), MBC-ATT={itr_mbc:.2f} (paper 17.19)")


if __name__ == "__main__":
    checks = [check_paper_dims, check_shapes, check_gradients_everywhere,
              check_cgscr_critical_path, check_coupling_changes_routing,
              check_routing_discrete_and_deterministic, check_morlet_init_in_band,
              check_logits_and_simplices, check_gate_per_sample,
              check_ablations_and_guards, check_overfit_tiny_batch,
              check_splits_and_normalizer, check_metrics]
    results = [c() for c in checks]
    print()
    if all(results):
        print(f">>> All {len(results)} checks PASSED.")
    else:
        print(">>> At least one check FAILED. Do NOT start a GPU run.")
        sys.exit(1)
