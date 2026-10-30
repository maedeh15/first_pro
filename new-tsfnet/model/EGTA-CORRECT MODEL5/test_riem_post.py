"""
test_riem_post.py -- --riem-head-fit post (priority-RIEMPOST). CPU, synthetic data, no GPU.

  1. before the head exists, the member model's output equals the plain trunk's (no blend): the trunk is trained and its
     checkpoint selected exactly as a plain run;
  2. after fit(), output == 0.5 * normalised trunk + 0.5 * head;
  3. a real train_one_fold call: the hook runs once, the head did NOT exist during training/selection, the checkpoint re-saved
     on disk contains the fitted head, and a fresh model loaded with strict=True reproduces the reported test accuracy.
"""
import os, tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import tsfnet_model_notebook as M


def main():
    torch.manual_seed(0); np.random.seed(0)
    D = 32
    plain = M.TSFNet().eval()
    post = M.TSFNet(riem_branch="member", riem_dim=D, riem_head_fit="post").eval()
    post.load_state_dict(plain.state_dict(), strict=False)
    xe, xf, xc = torch.randn(2, 16, 16, 600), torch.randn(2, 21, 16, 16, 30, 2), torch.randn(2, D)
    with torch.no_grad():
        yp = plain(xe, xf)[3]; yq = post(xe, xf, None, xc)[3]
    assert torch.allclose(yp, yq, atol=1e-6)
    print("[1] before fit: output == the plain trunk's (no blend) -> OK")

    rng = np.random.RandomState(0); X = rng.randn(120, D); y = rng.randint(0, 3, 120); X[:, 0] += y
    post.riem_head.fit(X, y)
    with torch.no_grad():
        yq2 = post(xe, xf, None, xc)[3]
    exp = 0.5 * (yp / (yp.sum(-1, keepdim=True) + 1e-8)) + 0.5 * post.riem_head(xc)
    assert torch.allclose(yq2, exp, atol=1e-6)
    print("[2] after fit: output == 0.5*normalised trunk + 0.5*head -> OK")

    def mk(n):
        yy = torch.arange(n) % 3
        return TensorDataset(torch.randn(n, 16, 16, 600), torch.randn(n, 21, 16, 16, 30, 2), yy,
                             torch.zeros(n, 0), torch.randn(n, D) + yy[:, None].float())
    tr, va, te = mk(36), mk(12), mk(12)
    ld = lambda d, sh: DataLoader(d, batch_size=12, shuffle=sh)
    model = M.TSFNet(riem_branch="member", riem_dim=D, riem_head_fit="post")
    calls = []

    def hook(m):
        assert float(m.riem_head.fitted) == 0.0, "the head must not exist during training/selection"
        m.riem_head.fit(torch.cat([tr.tensors[4], va.tensors[4]]), torch.cat([tr.tensors[2], va.tensors[2]]))
        calls.append(1)

    ck = os.path.join(tempfile.mkdtemp(), "VP_session1_best.pt")
    acc, _k, _m, _r = M.train_one_fold(model, ld(tr, True), ld(va, False), ld(te, False), torch.device("cpu"),
                                        ckpt_path=ck, normalize_ypred=True, no_decay_scalars=True,
                                        stage2_stop_mode="none", stage1_max_epochs=2, stage1_patience=5,
                                        log_fn=lambda *a, **k: None, post_select_hook=hook)
    assert calls == [1]
    sd = torch.load(ck)
    assert float(sd["riem_head.fitted"]) == 1.0 and float(sd["riem_head.W"].abs().sum()) > 0
    m2 = M.TSFNet(riem_branch="member", riem_dim=D, riem_head_fit="post"); m2.load_state_dict(sd, strict=True); m2.eval()
    ok = 0
    with torch.no_grad():
        for b in ld(te, False):
            ok += (m2(b[0], b[1], None, b[4])[3].argmax(1) == b[2]).sum().item()
    assert abs(ok / 12 - acc) < 1e-9, (ok / 12, acc)
    print(f"[3] train_one_fold: hook ran once, saved checkpoint holds the fitted head, strict replay "
          f"{ok / 12:.4f} == reported {acc:.4f} -> OK")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
