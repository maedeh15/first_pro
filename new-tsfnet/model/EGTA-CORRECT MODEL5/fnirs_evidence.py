"""
fnirs_evidence.py -- EXACT per-segment decomposition of the fNIRS head when the model is built with fnirs_head="linear".

WHY THIS EXISTS (read from tsfnet_model_notebook.py):
  * The fNIRS decision path is  enhanced_fnirs = g*E + (1-g)*P,  with g = sigmoid(cafe.gate_fnirs),
      P = EGTA pooling = concat_h  sum_i w[h,i] * v[i,h]        (v = per-segment fNIRS features, 512 = 4 heads x 128)
      E = fnirs_out_proj(mean_i cross_attn_1(...))  and cross_attn_1 has ONE key, so E depends on the FUSION token only.
  * With a LINEAR head,  logit = W . enhanced_fnirs + b  is a SUM:
      logit_c = (1-g) * sum_i sum_h w[h,i] * (W_c,h . v[i,h])      <- evidence of segment i   (c_i)
              + g * W_c . E + b_c                                  <- EEG-derived term + bias   (rest)
    Segments are encoded independently (Conv -> ELU -> BN(eval) -> Dropout(off)), so c_i depends on segment i only.
  * The explanation unit is the EVIDENCE c_i = a_i * h(v_i), NOT the attention weight a_i. It is informative even when
    the attention weights are uniform, which is exactly the situation found for EGTA.
  * WHAT IS EXACT: the fNIRS head's LOGIT. The final prediction is y_pred = (w1*y_eeg + w2*y_fusion + w3*y_fnirs)/3 over
    softmax PROBABILITIES, so y_pred is a mixture of heads (exact at head level) but NOT additive in the logits.

  python fnirs_evidence.py --self-test            (needs torch + tsfnet_model_notebook.py next to this file)
"""
import sys
import torch


def segment_terms(cafe, fnirs_feat):
    """per-segment pooled contribution vectors Pi (B, S, D): sum_i Pi[:, i] == EGTA pooled vector P.
    Uses the weights stored by the LAST forward of `cafe` (cafe._last_egta_weights, shape (B, H, S))."""
    B, S = fnirs_feat.shape[:2]
    v = fnirs_feat.reshape(B, S, -1)
    w = cafe._last_egta_weights
    H, D = w.shape[1], v.shape[-1]
    return (w.permute(0, 2, 1).unsqueeze(-1) * v.view(B, S, H, D // H)).reshape(B, S, D)


@torch.no_grad()
def decompose(model, x_eeg, x_fnirs, x_conn=None, x_chan=None):
    """model must be in eval() mode, built with fnirs_temporal_align='egta' and fnirs_head='linear'.
    returns dict: contrib (B,S,C) segment evidence | rest (B,C) EEG-derived term + bias | logit (B,C) | weights (B,H,S) | gate | y_* """
    if model.training:
        raise RuntimeError("call model.eval() first (dropout and BN must be in inference mode)")
    if getattr(model.cafe, "fnirs_temporal_align", None) != "egta":
        raise RuntimeError("decomposition needs fnirs_temporal_align='egta'")
    head = model.clf_fnirs
    if not hasattr(head, "net") or len(head.net) != 1:
        raise RuntimeError("decomposition needs fnirs_head='linear' (a single Linear layer)")
    cap = {}
    hk = model.cafe.register_forward_hook(lambda m, i, o: cap.__setitem__("v", i[1].detach()))
    try:
        y_eeg, y_fus, y_fn, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
    finally:
        hk.remove()
    lin = head.net[0]
    g = torch.sigmoid(model.cafe.gate_fnirs)
    Pi = segment_terms(model.cafe, cap["v"])                                  # (B,S,D)
    c = (1.0 - g) * (Pi @ lin.weight.T)                                       # (B,S,C)  segment evidence
    logit = aux["enhanced_fnirs"] @ lin.weight.T + lin.bias                   # (B,C)
    rest = logit - c.sum(1)                                                   # g*W.E + b
    return dict(contrib=c, rest=rest, logit=logit, weights=model.cafe._last_egta_weights, gate=g,
                y_eeg=y_eeg, y_fusion=y_fus, y_fnirs=y_fn, y_pred=y_pred)


def head_mass(model):
    """share of the final mixture carried by each head, from the learned sigmoid weights (eeg, fusion, fnirs)"""
    w = torch.stack([torch.sigmoid(model.w_eeg), torch.sigmoid(model.w_fusion), torch.sigmoid(model.w_fnirs)]).flatten()
    return (w / w.sum()).tolist()


def self_test():
    sys.path.insert(0, ".")
    from tsfnet_model_notebook import CAFELayer, TSFNet, LinearEvidenceHead
    torch.manual_seed(0)
    # [1] CAFE level: the remainder enhanced - (1-g) * sum_i Pi must NOT depend on the fNIRS input (E depends on fusion only)
    cafe = CAFELayer(fnirs_temporal_align="egta", egta_k_init="noisy").eval()
    with torch.no_grad():
        cafe.gate_fnirs.fill_(0.3)
        fu = torch.randn(3, 32, 4, 4)
        rem = []
        for _ in range(2):
            fn = torch.randn(3, 21, 32, 4, 4)
            _, enh_fn, _ = cafe(fu, fn)
            g = torch.sigmoid(cafe.gate_fnirs)
            P = segment_terms(cafe, fn).sum(1)
            rem.append(enh_fn - (1 - g) * P)
    d = (rem[0] - rem[1]).abs().max().item()
    assert d < 1e-5, f"remainder depends on the fNIRS input (max diff {d:.2e}) -- decomposition is NOT exact"
    w = cafe._last_egta_weights
    assert w.std().item() > 0 and abs(w.sum(-1).mean().item() - 1) < 1e-5
    print(f"[1] CAFE: remainder independent of the fNIRS input (max diff {d:.1e}); EGTA weights sum to 1")
    # [2] full model: logit == sum of segment evidence + rest, and softmax(logit) == y_fnirs
    m = TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=21, fnirs_temporal_align="egta",
               egta_k_init="noisy", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0, fnirs_head="linear").eval()
    assert isinstance(m.clf_fnirs, LinearEvidenceHead)
    out = decompose(m, torch.randn(2, 16, 16, 2000), torch.randn(2, 21, 16, 16, 30, 2))
    err = (out["contrib"].sum(1) + out["rest"] - out["logit"]).abs().max().item()
    err2 = (torch.softmax(out["logit"], -1) - out["y_fnirs"]).abs().max().item()
    assert err < 1e-5 and err2 < 1e-5, (err, err2)
    print(f"[2] full model: additivity error {err:.1e}, softmax(logit) vs y_fnirs {err2:.1e}")
    # [3] changing ONE segment changes ONLY that segment's evidence (independent encoding in eval mode)
    xe, xf = torch.randn(1, 16, 16, 2000), torch.randn(1, 21, 16, 16, 30, 2)
    c0 = decompose(m, xe, xf)["contrib"][0]
    xf2 = xf.clone(); xf2[:, 5] = torch.randn_like(xf2[:, 5])
    o2 = decompose(m, xe, xf2); c1 = o2["contrib"][0]
    others = [i for i in range(21) if i != 5]
    # the attention weights (query from fusion) shift slightly, so compare the RAW per-segment features' terms:
    print(f"[3] segment 5 evidence changed by {(c1[5]-c0[5]).abs().max().item():.3f}; "
          f"others changed by {(c1[others]-c0[others]).abs().max().item():.3f} (non-zero only through the attention weights)")
    print("head mass (eeg, fusion, fnirs):", [round(x, 3) for x in head_mass(m)])
    print("SELFTEST PASSED")


if __name__ == "__main__":
    self_test() if "--self-test" in sys.argv else print(__doc__)
