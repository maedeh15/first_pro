"""
xai_trunk.py -- let the existing XAI scripts explain the TRUNK of a checkpoint that also carries a Riemannian head.

Checkpoints from --riem-branch member runs (gradient head, closed-form head, 'post', 'post_oof', or an assembled model) hold the
trunk's tensors under their usual names plus extra 'riem_head.*' tensors. The XAI scripts build a plain TSFNet (attention maps,
SHAP, Grad-CAM, faithfulness all concern the deep network), so loading such a checkpoint strict raises a key error. This helper
drops ONLY the 'riem_head.*' tensors and says so once, so every analysis then explains the deep half of the system.

What this means for interpretation (say it wherever the numbers are quoted):
  * the explanation is of the TRUNK (the deep network), not of the blended system whose accuracy is ~0.67;
  * the linear member is explained separately by xai_member_coefficients.py, and the split between the two by
    xai_component_attribution.py;
  * for 'post' / 'post_oof' / assembled checkpoints the trunk was trained and selected without any head (or against an honest
    one), so the explanation is of a network that never leaned on a memorised head. For the gradient-head model (84c18210ab)
    the trunk was trained jointly with the head.
"""
_ANNOUNCED = False


def load_checkpoint(path, map_location="cpu"):
    """torch.load without the 'weights_only=False' FutureWarning (and without unpickling arbitrary objects): a state_dict is tensors and
    plain python containers, which weights_only=True loads. If a checkpoint holds something else, fall back to the old behaviour, loudly."""
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:                                   # torch < 1.13 has no such argument
        return torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"  [xai] {path}: not loadable with weights_only=True ({type(e).__name__}); loading it the old way -- only do this for files you trust")
        return torch.load(path, map_location=map_location, weights_only=False)


def trunk_state(state):
    global _ANNOUNCED
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    drop = [k for k in state if k.startswith("riem_head.")]
    if drop and not _ANNOUNCED:
        print(f"  [xai] checkpoint carries a Riemannian head ({len(drop)} tensors, dropped): the explanations below are of the "
              f"TRUNK only; see xai_member_coefficients.py / xai_component_attribution.py for the member and the split.")
        _ANNOUNCED = True
    return {k: v for k, v in state.items() if not k.startswith("riem_head.")}


_XAI_ANNOUNCED = False


def set_xai_output(model):
    """Make the model return the ROW-NORMALISED y_pred (rows sum to 1), the quantity behind every reported accuracy / ensemble posterior.
    The raw y_pred of paper_mean fusion sums to (w_eeg + w_fusion + w_fnirs)/3, which is not 1, so explaining it explains a different
    quantity. Set TSFNET_XAI_RAW=1 in the environment to reproduce the earlier (raw) explanations. Training/eval code never calls this."""
    global _XAI_ANNOUNCED
    import os
    raw = os.environ.get("TSFNET_XAI_RAW", "0") == "1"
    model.xai_normalise_output = not raw
    if not _XAI_ANNOUNCED:
        print(f"  [xai] explaining the {'RAW' if raw else 'ROW-NORMALISED'} y_pred (TSFNET_XAI_RAW=1 selects the raw one)")
        _XAI_ANNOUNCED = True
    return model


def self_test():
    import torch
    sd = {"a.weight": torch.zeros(1), "riem_head.W": torch.zeros(3, 2), "riem_head.fitted": torch.ones(1)}
    out = trunk_state(sd)
    assert list(out) == ["a.weight"]
    assert list(trunk_state({"model_state_dict": sd})) == ["a.weight"]
    assert list(trunk_state({"a.weight": 1})) == ["a.weight"]          # a plain checkpoint passes through untouched
    import tempfile, os, warnings
    f = os.path.join(tempfile.mkdtemp(), "c.pt"); torch.save({"a.weight": torch.ones(2)}, f)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # the FutureWarning must be gone
        assert torch.equal(load_checkpoint(f)["a.weight"], torch.ones(2))
    print("xai_trunk self-test passed")


if __name__ == "__main__":
    self_test()
