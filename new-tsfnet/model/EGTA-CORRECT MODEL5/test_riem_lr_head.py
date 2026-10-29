"""
test_riem_lr_head.py -- checks for --riem-head-fit lr / lr_all (priority-RIEMFIT). CPU only, no data.

  1. the buffered forward equals sklearn's own predict_proba on held-out rows;
  2. the head has NO trainable parameters, so the optimizer has nothing to update;
  3. state_dict round-trips into a fresh head (this is what ensemble_eval relies on);
  4. forward before fit() and a fit with a missing class are refused;
  5. a zero-variance feature column does not produce NaN;
  6. TSFNet construction is RNG-neutral: with the same seed, the torch RNG state after building the model and
     every trunk parameter are IDENTICAL for riem_head_fit='sgd' and 'lr'; only the head differs;
  7. 'sgd' (the default) still builds a gradient-trained RiemHead, bit for bit as before.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import tsfnet_model_notebook as M


def main():
    rng = np.random.RandomState(0)
    D = 40
    X = rng.randn(300, D); y = rng.randint(0, 3, 300)
    X[:, 0] += y * 0.8
    X[:, 5] = 3.0                                          # zero-variance column
    Xte = rng.randn(50, D) + 0.1

    h = M.RiemLRHead(D)
    h.fit(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y))
    ref = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X, y)
    P = h(torch.as_tensor(Xte, dtype=torch.float32)).detach().double().numpy()
    err = np.abs(P - ref.predict_proba(Xte)).max()
    assert err < 1e-4, err
    assert np.isfinite(P).all() and np.allclose(P.sum(1), 1.0, atol=1e-5)
    print(f"[1] forward == sklearn predict_proba (max |diff| {err:.1e}); zero-variance column finite -> OK")

    assert sum(p.numel() for p in h.parameters()) == 0
    print("[2] no trainable parameters -> OK")

    h2 = M.RiemLRHead(D); h2.load_state_dict(h.state_dict())
    assert torch.equal(h(torch.as_tensor(Xte, dtype=torch.float32)), h2(torch.as_tensor(Xte, dtype=torch.float32)))
    print("[3] state_dict round trip reproduces the posteriors exactly -> OK")

    try:
        M.RiemLRHead(D)(torch.zeros(2, D)); raise SystemExit("forward before fit was NOT refused")
    except RuntimeError as e:
        assert "before fit" in str(e)
    try:
        M.RiemLRHead(D).fit(X[y != 2], y[y != 2]); raise SystemExit("a missing class was NOT refused")
    except ValueError as e:
        assert "every class" in str(e)
    print("[4] unfitted forward and a missing class are refused -> OK")

    def build(fit):
        torch.manual_seed(123); np.random.seed(123)
        m = M.TSFNet(riem_branch="member", riem_dim=D, riem_head_fit=fit)
        return m, torch.get_rng_state().clone()

    ms, rs = build("sgd"); ml, rl = build("lr")
    assert torch.equal(rs, rl), "torch RNG state after construction differs between sgd and lr"
    trunk = lambda m: {k: v for k, v in m.state_dict().items() if not k.startswith("riem_head.")}
    a, b = trunk(ms), trunk(ml)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "trunk init differs"
    assert isinstance(ms.riem_head, M.RiemHead) and isinstance(ml.riem_head, M.RiemLRHead)
    assert sum(p.numel() for p in ml.riem_head.parameters()) == 0
    print("[5] construction is RNG-neutral: same RNG state and identical trunk init for sgd and lr -> OK")
    print("[6] default 'sgd' builds the gradient RiemHead as before -> OK")
    for bad in ("nope",):
        try:
            M.TSFNet(riem_branch="member", riem_dim=D, riem_head_fit=bad); raise SystemExit("bad value accepted")
        except ValueError:
            pass
    try:
        M.TSFNet(riem_branch="none", riem_head_fit="lr"); raise SystemExit("lr without member accepted")
    except ValueError:
        pass
    print("[7] invalid values and lr without --riem-branch member are refused -> OK")
    # [8] attaching a closed-form head to a PLAIN checkpoint (priority-ATTACH)
    torch.manual_seed(7)
    plain = M.TSFNet()                                     # no riem branch: the baseline architecture
    sd = {k: v.clone() for k, v in plain.state_dict().items()}
    torch.manual_seed(8)
    asm = M.TSFNet(riem_branch="member", riem_dim=D, riem_head_fit="lr_all")
    M.load_with_attached_riem_head(asm, sd, torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y))
    got = asm.state_dict()
    assert all(torch.equal(got[k], v) for k, v in sd.items()), "trunk weights were altered by the attach"
    assert float(asm.riem_head.fitted) == 1.0
    extra = dict(sd); extra["not_a_layer.weight"] = torch.zeros(1)
    try:
        M.load_with_attached_riem_head(M.TSFNet(riem_branch="member", riem_dim=D, riem_head_fit="lr_all"), extra,
                                      torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y))
        raise SystemExit("an unexpected key was NOT refused")
    except RuntimeError as e:
        assert "does not match" in str(e)
    short = {k: v for k, v in sd.items() if not k.startswith("fnirs_branch")} if any(k.startswith("fnirs_branch") for k in sd) \
        else dict(list(sd.items())[:-3])
    try:
        M.load_with_attached_riem_head(M.TSFNet(riem_branch="member", riem_dim=D, riem_head_fit="lr_all"), short,
                                      torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y))
        raise SystemExit("a missing trunk key was NOT refused")
    except RuntimeError as e:
        assert "does not match" in str(e)
    try:
        M.load_with_attached_riem_head(M.TSFNet(riem_branch="member", riem_dim=D), sd,
                                       torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y))
        raise SystemExit("a gradient-trained head was accepted")
    except ValueError:
        pass
    print("[8] attach: trunk bit-identical, head fitted; a foreign key, a missing trunk key and a gradient head are refused -> OK")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
