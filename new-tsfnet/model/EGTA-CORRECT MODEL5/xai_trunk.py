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


def self_test():
    import torch
    sd = {"a.weight": torch.zeros(1), "riem_head.W": torch.zeros(3, 2), "riem_head.fitted": torch.ones(1)}
    out = trunk_state(sd)
    assert list(out) == ["a.weight"]
    assert list(trunk_state({"model_state_dict": sd})) == ["a.weight"]
    assert list(trunk_state({"a.weight": 1})) == ["a.weight"]          # a plain checkpoint passes through untouched
    print("xai_trunk self-test passed")


if __name__ == "__main__":
    self_test()
