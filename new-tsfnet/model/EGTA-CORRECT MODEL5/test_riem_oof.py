"""
test_riem_oof.py -- --riem-head-fit post_oof (priority-RIEMOOF). CPU, synthetic data.

  1. oof_head_posteriors: every row is scored by a head that never saw its own session (checked by construction: a label
     pattern that exists in ONE session only cannot be recovered), rows sum to 1, a single training session is refused;
  2. forward in the training phase uses the three extra columns as the head's posterior (0.5*normalised trunk + 0.5*columns)
     and refuses a feature tensor without them; after fit() the columns are ignored and the real head is used;
  3. train_one_fold end to end with width D+3 features: the hook runs once, the checkpoint holds the fitted head, and a fresh
     model loaded strict and fed plain width-D features reproduces the reported test accuracy.
"""
import os, tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import tsfnet_model_notebook as M


def main():
    torch.manual_seed(0); rng = np.random.RandomState(0)
    D = 32
    # 1) out-of-session posteriors
    n = 90
    y = np.tile([0, 1, 2], n // 3); ses = np.repeat([1, 2], n // 2)
    F = rng.randn(n, D)
    F[ses == 1, 0] += 3.0 * (y[ses == 1] == 1)          # class 1 is visible in session 1 ONLY ...
    P = M.oof_head_posteriors(F, y, ses)
    assert P.shape == (n, 3) and np.allclose(P.sum(1), 1.0, atol=1e-6)
    acc2 = (P[ses == 2].argmax(1) == y[ses == 2]).mean()   # ... so a head fitted on session 1 cannot score session 2 well
    assert acc2 < 0.55, acc2
    try:
        M.oof_head_posteriors(F[ses == 1], y[ses == 1], ses[ses == 1]); raise SystemExit("one session was NOT refused")
    except ValueError:
        pass
    print(f"[1] out-of-session posteriors: rows sum to 1, a session-specific pattern is not leaked (acc {acc2:.2f}), "
          f"one training session refused -> OK")

    # 2) forward
    plain = M.TSFNet().eval()
    m = M.TSFNet(riem_branch="member", riem_dim=D, riem_head_fit="post_oof").eval()
    m.load_state_dict(plain.state_dict(), strict=False)
    xe, xf = torch.randn(2, 16, 16, 600), torch.randn(2, 21, 16, 16, 30, 2)
    feats = torch.randn(2, D); cols = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    with torch.no_grad():
        yp = plain(xe, xf)[3]
        y1 = m(xe, xf, None, torch.cat([feats, cols], 1))[3]
    exp = 0.5 * (yp / (yp.sum(-1, keepdim=True) + 1e-8)) + 0.5 * cols
    assert torch.allclose(y1, exp, atol=1e-6)
    try:
        m(xe, xf, None, feats); raise SystemExit("features without the OOF columns were NOT refused")
    except ValueError:
        pass
    X = rng.randn(120, D); yy = rng.randint(0, 3, 120); X[:, 0] += yy
    m.riem_head.fit(X, yy)
    with torch.no_grad():
        y2 = m(xe, xf, None, torch.cat([feats, cols], 1))[3]
        y3 = m(xe, xf, None, feats)[3]
    assert torch.allclose(y2, y3, atol=1e-6), "after fit() the extra columns must be ignored"
    print("[2] training phase blends the carried columns; after fit() they are ignored and the real head is used -> OK")

    # 3) end to end
    def mk(nn_, extra):
        t = torch.arange(nn_) % 3
        f = torch.randn(nn_, D) + t[:, None].float()
        if extra:
            f = torch.cat([f, torch.softmax(torch.randn(nn_, 3), 1)], 1)
        return TensorDataset(torch.randn(nn_, 16, 16, 600), torch.randn(nn_, 21, 16, 16, 30, 2), t, torch.zeros(nn_, 0), f), t, f
    tr, ttr, ftr = mk(36, True); va, tva, fva = mk(12, True); te, tte, fte = mk(12, True)
    ld = lambda d, sh: DataLoader(d, batch_size=12, shuffle=sh)
    model = M.TSFNet(riem_branch="member", riem_dim=D, riem_head_fit="post_oof")
    calls = []

    def hook(mm):
        assert float(mm.riem_head.fitted) == 0.0
        mm.riem_head.fit(torch.cat([ftr[:, :D], fva[:, :D]]), torch.cat([ttr, tva])); calls.append(1)

    ck = os.path.join(tempfile.mkdtemp(), "VP_session1_best.pt")
    acc, _k, _m, _r = M.train_one_fold(model, ld(tr, True), ld(va, False), ld(te, False), torch.device("cpu"),
                                        ckpt_path=ck, normalize_ypred=True, no_decay_scalars=True, stage2_stop_mode="none",
                                        stage1_max_epochs=2, stage1_patience=5, log_fn=lambda *a, **k: None,
                                        post_select_hook=hook)
    assert calls == [1]
    sd = torch.load(ck); assert float(sd["riem_head.fitted"]) == 1.0
    m2 = M.TSFNet(riem_branch="member", riem_dim=D, riem_head_fit="post_oof"); m2.load_state_dict(sd, strict=True); m2.eval()
    ok = 0
    with torch.no_grad():
        for b, ff in zip(ld(te, False), ld(TensorDataset(fte[:, :D]), False)):
            ok += (m2(b[0], b[1], None, ff[0])[3].argmax(1) == b[2]).sum().item()
    assert abs(ok / 12 - acc) < 1e-9, (ok / 12, acc)
    print(f"[3] train_one_fold with width D+3 features: hook once, head saved, strict replay on plain width-D features "
          f"{ok / 12:.4f} == reported {acc:.4f} -> OK")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
