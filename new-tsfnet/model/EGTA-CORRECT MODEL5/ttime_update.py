"""
ttime_update.py -- the target-model-update half of T-TIME (Li et al., IEEE TBME), as reusable
pieces: the loss, the temperature rescaling, and the online schedule. ensemble_eval.py --ttime
wires these to the real model; keeping them here makes the schedule and the loss unit-testable
without a GPU, which is the part where a silent mistake would be invisible in the final number.

Reference implementation: sylyoung/DeepTransferEEG, tl/ttime.py. Two faithful-on-purpose details
and three deliberate deviations, all of which change what the result means:

FAITHFUL
  * loss = conditional entropy minimisation + marginal distribution regularisation, on a sliding
    batch of the most recent B test windows, with temperature T on the probabilities.
  * online schedule: the prediction for window a is made BEFORE any update that uses window a, so
    no future window can influence it. Their loop does the same (predict, then update).

DEVIATIONS
  * Their loss is on softmax(logits / T). This model's heads already return NORMALISED
    probabilities (--normalize-ypred), and the logits are not exposed, so temperature is applied
    as p^(1/T) renormalised, which is exactly softmax(log p / T). temper() asserts this identity.
  * They call model.train() during the update. Here the model stays in eval() by default: this
    project's baseline applies AdaBN first, and train-mode BatchNorm would overwrite those
    adapted running statistics from batches of 8 and re-enable two dropout layers, confounding
    the update with the loss of AdaBN. --ttime-train-mode train restores their behaviour.
  * No incremental Euclidean alignment. Their IEA whitens the raw input per test trial; input-level
    covariance alignment is a SEPARATE variable in this project (covEA, already tested and
    rejected), so adding it here would make this a two-variable experiment.

The update uses no labels, but it uses the unlabeled test session, so it is transductive in the
same way as AdaBN and per-session normalisation -- disclose it with them, not as a separate claim.
"""
import numpy as np
import torch

EPS = 1e-5


def temper(p, T, eps=1e-12):
    """p^(1/T) renormalised == softmax(log p / T). p: (..., K) probabilities summing to 1."""
    if T == 1.0:
        return p
    logp = torch.log(p.clamp_min(eps))
    return torch.softmax(logp / T, dim=-1)


def cem_mdr_loss(p, T=2.0, mdr="balanced", zk=None, c=4.0, eps=EPS):
    """T-TIME's update loss on one sliding batch.

    p    : (B, K) normalised probabilities from the head being adapted
    mdr  : "balanced" = Eq. 13 on the batch mean; "adaptive" = Eq. 10-13, the batch mean divided
           by (c + the running pseudo-label count zk) before normalising, which stops a
           class-imbalanced batch from being pushed towards uniform; "off" = CEM alone, which the
           paper's own ablation shows collapses towards one class.
    Returns (loss, batch_mean_probabilities)."""
    q = temper(p, T)
    cem = -(q * torch.log(q + eps)).sum(dim=1).mean()          # Eq. 8
    pbar = q.mean(dim=0)                                        # Eq. 9
    if mdr == "off":
        return cem, pbar
    if mdr == "balanced":
        reg = (pbar * torch.log(pbar + eps)).sum()              # Eq. 13, == -H(pbar)
    elif mdr == "adaptive":
        if zk is None:
            raise ValueError("mdr='adaptive' needs zk, the running pseudo-label counts")
        z = torch.as_tensor(zk, dtype=pbar.dtype, device=pbar.device)
        qk = pbar / (c + z)                                     # Eq. 11
        qk = qk / qk.sum()                                      # Eq. 12
        reg = (qk * torch.log(qk + eps)).sum()                  # Eq. 13
    else:
        raise ValueError(f"mdr must be balanced/adaptive/off, got {mdr!r}")
    return cem + reg, pbar


def online_schedule(n, batch, stride=1):
    """The paper's order of operations, as (index_to_predict, batch_slice_or_None) pairs.

    For each window a in 0..n-1: predict a, then -- only if at least `batch` windows have arrived
    and a is on the stride -- update on windows [a-batch+1, a]. The prediction for a is therefore
    produced by a model updated on windows < a only. With n < batch no update ever happens and the
    model stays exactly as AdaBN left it, so the run degenerates to the baseline rather than to
    something half-adapted."""
    if batch < 1 or stride < 1:
        raise ValueError("batch and stride must be >= 1")
    for a in range(n):
        upd = None
        if (a + 1) >= batch and (a + 1) % stride == 0:
            upd = slice(a - batch + 1, a + 1)
        yield a, upd


def update_pseudo_counts(zk, p, tau):
    """Eq. 10: count windows whose top probability clears tau, per class, accumulated over the
    session. Returns a new array; never mutates the input."""
    zk = np.asarray(zk, dtype=np.float64).copy()
    pm = p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
    conf = pm.max(axis=1)
    cls = pm.argmax(axis=1)
    for k in np.unique(cls[conf >= tau]):
        zk[k] += int(((cls == k) & (conf >= tau)).sum())
    return zk


def trainable_parameters(model, scope="all"):
    """scope 'all' = every parameter (the paper, which argues BN-only is unstable);
    'bn' = affine parameters of normalisation layers only (Tent's scope, for comparison)."""
    if scope == "all":
        return [q for q in model.parameters() if q.requires_grad]
    if scope != "bn":
        raise ValueError(f"scope must be all/bn, got {scope!r}")
    out = []
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                          torch.nn.GroupNorm, torch.nn.LayerNorm)):
            out += [q for q in (m.weight, m.bias) if q is not None and q.requires_grad]
    if not out:
        raise ValueError("scope='bn' found no normalisation parameters to update")
    return out


# ---------------------------------------------------------------------------- self-test
def _self_test():
    torch.manual_seed(0)
    # temper() is exactly softmax(log p / T)
    logits = torch.randn(7, 3)
    p = torch.softmax(logits, -1)
    for T in (1.0, 2.0, 0.5, 4.0):
        assert torch.allclose(temper(p, T), torch.softmax(logits / T, -1), atol=1e-6), T
    print("1) temper(p,T) == softmax(logits/T) for T in 1, 2, 0.5, 4")

    # CEM falls when predictions sharpen; MDR punishes a collapsed marginal
    sharp = torch.tensor([[0.98, 0.01, 0.01]] * 6)
    flat = torch.full((6, 3), 1 / 3)
    c_sharp, _ = cem_mdr_loss(sharp, mdr="off")
    c_flat, _ = cem_mdr_loss(flat, mdr="off")
    assert c_sharp < c_flat, (c_sharp, c_flat)
    # all six windows on one class: balanced MDR must prefer the spread-out batch
    spread = torch.tensor([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]] * 2)
    l_coll, _ = cem_mdr_loss(sharp, mdr="balanced")
    l_spread, _ = cem_mdr_loss(spread, mdr="balanced")
    assert l_spread < l_coll, (l_spread, l_coll)
    print(f"2) CEM rewards confidence ({c_sharp:.3f} < {c_flat:.3f}); MDR rewards a spread marginal"
          f" ({l_spread:.3f} < {l_coll:.3f})")

    # adaptive MDR: once class 0 dominates the pseudo-label counts, a batch that is ALSO all
    # class 0 is penalised less than under the balanced form (that is the point of Eq. 11)
    zk = np.array([100.0, 0.0, 0.0])
    l_bal, _ = cem_mdr_loss(sharp, mdr="balanced")
    l_ad, _ = cem_mdr_loss(sharp, mdr="adaptive", zk=zk, c=4.0)
    assert l_ad < l_bal, (l_ad, l_bal)
    print(f"3) adaptive MDR tolerates an observed class skew ({l_ad:.3f} < {l_bal:.3f})")

    # the online schedule can never use a future window for the current prediction
    n, B = 20, 8
    for a, upd in online_schedule(n, B):
        if upd is not None:
            assert upd.stop == a + 1 and upd.start == a - B + 1
    first = [a for a, u in online_schedule(n, B) if u is not None]
    assert first[0] == B - 1 and len(first) == n - B + 1
    assert all(u is None for _, u in online_schedule(5, B)), "n < batch must never update"
    strided = [a for a, u in online_schedule(n, B, stride=4) if u is not None]
    assert strided == [7, 11, 15, 19], strided   # a+1 == 8 is both >= batch and on the stride
    print(f"4) online schedule: first update at window {first[0]}, {len(first)} updates, none "
          f"reaching past the predicted window; n<batch does nothing; stride=4 -> {strided}")

    # pseudo-label counts: only confident windows count, input untouched
    pp = torch.tensor([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25], [0.05, 0.9, 0.05]])
    z0 = np.zeros(3)
    z1 = update_pseudo_counts(z0, pp, tau=0.7)
    assert z1.tolist() == [1.0, 1.0, 0.0] and z0.tolist() == [0.0, 0.0, 0.0]
    print("5) pseudo-label counts take only windows above tau and do not mutate the input")

    # parameter scopes
    net = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 3), torch.nn.BatchNorm3d(2), torch.nn.Linear(2, 2))
    assert len(trainable_parameters(net, "all")) == 6
    assert len(trainable_parameters(net, "bn")) == 2
    try:
        trainable_parameters(torch.nn.Sequential(torch.nn.Linear(2, 2)), "bn")
        raise SystemExit("scope='bn' with no norm layer must raise")
    except ValueError:
        pass
    print("6) parameter scopes: all=6, bn=2, and bn with no norm layer is refused")

    # end to end on a toy model: one step of the loss must reduce the batch's mean entropy
    tiny = torch.nn.Linear(4, 3)
    x = torch.randn(8, 4)
    opt = torch.optim.Adam(tiny.parameters(), lr=0.05)
    ent = lambda: float(-(torch.softmax(tiny(x), -1) * torch.log_softmax(tiny(x), -1)).sum(1).mean())
    e0 = ent()
    for _ in range(20):
        loss, _ = cem_mdr_loss(torch.softmax(tiny(x), -1), T=2.0, mdr="balanced")
        opt.zero_grad(); loss.backward(); opt.step()
    e1 = ent()
    assert e1 < e0 - 0.05, (e0, e1)
    print(f"7) 20 steps on a toy head cut the batch's mean entropy {e0:.3f} -> {e1:.3f}")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    _self_test()
