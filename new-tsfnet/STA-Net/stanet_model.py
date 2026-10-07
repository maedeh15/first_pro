# ============================================================================
# STA-Net (PyTorch re-implementation, faithful to the OFFICIAL TensorFlow code
# https://github.com/MutianLiu-SHU/STA-Net/blob/main/sta.py)
#   Liu et al., "STA-Net: Spatial-temporal alignment network for hybrid EEG-fNIRS
#   decoding", Information Fusion 119 (2025) 103023.
#
# Everything that the paper text leaves ambiguous follows sta.py:
#   * conv -> BatchNorm -> ELU (BN eps=1e-3, momentum 0.99 == torch momentum 0.01)
#   * Keras glorot-uniform init, HeUniform positional embedding (1, l, 512)
#   * the fNIRS conv is shared over the l=11 segments (they act as extra batch dims)
#   * gamma residual = sigmoid(w), w initialised at 0  (so gamma starts at 0.5)
#   * Keras MultiHeadAttention(num_heads=10, key_dim=256): Q/K/V are projected to
#     heads*key_dim, the output is projected back to 256 (NOT torch's nn.MultiheadAttention)
#   * loss = CE(final) + CE(eeg) + (1-PCC_fgsa1) + (1-PCC_fgsa2) + (1-PCC_egta)  (sum, as in code)
# Input layout (as saved by Preproc_stanet_nback.py):
#   eeg   : (B, 16, 16, 600)  or (B, 16, 16, 600, 1)
#   fnirs : (B, l, 16, 16, 30, 2)         (already multiplied by 1e3, see run script)
# ============================================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pearson_r(a, b):
    """Official pearson_r: PCC along axis 1 per sample (population std, +1e-6), |.|, mean over batch."""
    am = a - a.mean(dim=1, keepdim=True)
    bm = b - b.mean(dim=1, keepdim=True)
    num = (am * bm).mean(dim=1)
    den = a.std(dim=1, unbiased=False) * b.std(dim=1, unbiased=False) + 1e-6
    return (num / den).abs().mean()


def _glorot_(w):
    if w.dim() >= 3:   # conv kernels (out, in, k1, k2, k3): Keras fans use receptive-field size
        rf = int(torch.tensor(w.shape[2:]).prod())
        fan_in, fan_out = w.shape[1] * rf, w.shape[0] * rf
    else:
        fan_out, fan_in = w.shape
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(w, -limit, limit)


class Conv3dSame(nn.Module):
    """TF padding='same' (extra pad goes to the end) for any stride."""

    def __init__(self, cin, cout, k, s):
        super().__init__()
        self.k, self.s = tuple(k), tuple(s)
        self.conv = nn.Conv3d(cin, cout, self.k, stride=self.s, padding=0)
        _glorot_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        pads = []
        for dim, k, s in zip(x.shape[-3:], self.k, self.s):
            out = -(-dim // s)
            need = max((out - 1) * s + k - dim, 0)
            pads = [need // 2, need - need // 2] + pads
        return self.conv(F.pad(x, pads))


def _bn(c):
    return nn.BatchNorm3d(c, eps=1e-3, momentum=0.01)   # keras: epsilon=1e-3, momentum=0.99


def _dense(i, o):
    l = nn.Linear(i, o)
    _glorot_(l.weight)
    nn.init.zeros_(l.bias)
    return l


class FGSA(nn.Module):
    """sta.py `fga`. eeg_fusion/eeg: (B,C,H,W,T); fnirs: (B,S,C,H',W',T'). Returns feature, PCC."""

    def __init__(self, c_fnirs, tem_k):
        super().__init__()
        self.pool = Conv3dSame(c_fnirs, 1, (3, 3, tem_k), (1, 1, 1))
        self.residual_para = nn.Parameter(torch.zeros(()))

    def forward(self, eeg_fusion, eeg, fnirs):
        B, S = fnirs.shape[:2]
        a = self.pool(fnirs.flatten(0, 1))                 # (B*S,1,H,W,T')
        a = a.mean(dim=-1, keepdim=True)                   # gap over time
        a = a.view(B, S, *a.shape[1:]).mean(dim=1)         # mean over segments -> (B,1,H,W,1)
        phi = torch.sigmoid(a)
        guided = eeg_fusion * phi
        g = torch.sigmoid(self.residual_para)
        out = guided + (g * eeg + (1 - g) * eeg_fusion)
        pcc = pearson_r(eeg.mean(dim=(-1, 1)).flatten(1),   # official: mean over time & channels
                        phi.flatten(1))
        return out, pcc


class ConvBlock(nn.Module):
    def __init__(self, c_in_eeg, c_in_fn, c_out, eeg_k, eeg_s, fn_k, fn_s, tem_k):
        super().__init__()
        self.eeg_conv, self.eeg_bn = Conv3dSame(c_in_eeg, c_out, eeg_k, eeg_s), _bn(c_out)
        self.fn_conv, self.fn_bn = Conv3dSame(c_in_fn, c_out, fn_k, fn_s), _bn(c_out)
        self.fu_conv, self.fu_bn = Conv3dSame(c_in_eeg, c_out, eeg_k, eeg_s), _bn(c_out)
        self.fga = FGSA(c_out, tem_k)

    def forward(self, fusion, eeg, fnirs):
        B, S = fnirs.shape[:2]
        e = F.elu(self.eeg_bn(self.eeg_conv(eeg)))
        f = F.elu(self.fn_bn(self.fn_conv(fnirs.flatten(0, 1))))   # BN stats over B*S, as Keras
        f = f.view(B, S, *f.shape[1:])
        u = F.elu(self.fu_bn(self.fu_conv(fusion)))
        u, pcc = self.fga(u, e, f)
        return u, e, f, pcc


class EGTA(nn.Module):
    """sta.py `e_f_attention` (emb 256, key_dim 256, 10 heads, dropout 0.5 on attention probs)."""

    def __init__(self, in_dim=512, seg_len=11, emb=256, key_dim=256, heads=10, drop=0.5):
        super().__init__()
        self.heads, self.key_dim, self.emb = heads, key_dim, emb
        self.q_proj, self.fusion_proj, self.k_proj = _dense(in_dim, emb), _dense(in_dim, emb), _dense(in_dim, emb)
        self.pos = nn.Parameter(torch.empty(1, seg_len, in_dim))
        nn.init.uniform_(self.pos, -math.sqrt(6.0 / seg_len), math.sqrt(6.0 / seg_len))   # keras HeUniform fan_in=shape[-2]
        self.wq, self.wk, self.wv = (_dense(emb, heads * key_dim) for _ in range(3))
        self.wo = _dense(heads * key_dim, emb)
        self.attn_drop = nn.Dropout(drop)

    def forward(self, eeg_fusion, fnirs_seq):
        # eeg_fusion: (B,512)   fnirs_seq: (B,l,512)
        fusion_out = self.fusion_proj(eeg_fusion)
        q = self.q_proj(eeg_fusion).unsqueeze(1)                     # (B,1,256)
        k = self.k_proj(fnirs_seq + self.pos)                        # (B,l,256)
        B, l, _ = k.shape
        Q = self.wq(q).view(B, 1, self.heads, self.key_dim).transpose(1, 2)
        K = self.wk(k).view(B, l, self.heads, self.key_dim).transpose(1, 2)
        V = self.wv(k).view(B, l, self.heads, self.key_dim).transpose(1, 2)
        att = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(self.key_dim), dim=-1)   # (B,H,1,l)
        out = (self.attn_drop(att) @ V).transpose(1, 2).reshape(B, 1, self.heads * self.key_dim)
        fn_w = self.wo(out).squeeze(1)                               # (B,256)
        pcc = pearson_r(q.squeeze(1), fn_w)
        return fusion_out, fn_w, att.mean(dim=(1, 2)), pcc           # attention weights (B,l)


class STANet(nn.Module):
    def __init__(self, num_classes=3, seg_len=11, drop=0.5):
        super().__init__()
        self.b1 = ConvBlock(1, 2, 16, (2, 2, 13), (2, 2, 6), (2, 2, 5), (2, 2, 2), 5)
        self.b2 = ConvBlock(16, 16, 32, (2, 2, 5), (2, 2, 2), (2, 2, 3), (2, 2, 2), 3)
        self.drop = nn.Dropout(drop)
        self.egta = EGTA(512, seg_len, 256, 256, 10, drop)
        self.fu_fc, self.fn_fc, self.eeg_fc = _dense(256, 256), _dense(256, 256), _dense(512, 256)
        self.fu_out, self.fn_out, self.eeg_out = _dense(256, num_classes), _dense(256, num_classes), _dense(256, num_classes)
        self.fu_w, self.fn_w = _dense(256, 1), _dense(256, 1)

    @staticmethod
    def _flat_cl(x):                 # (B,C,H,W,1) -> keras channels-last flatten (B, H*W*1*C)
        return x.permute(0, 2, 3, 4, 1).flatten(1)

    def forward(self, eeg, fnirs):
        """returns dict: p_final, p_eeg, p_fusion, p_fnirs (probabilities), w (B,2), att (B,l), reg (scalar)"""
        if eeg.dim() == 4:
            eeg = eeg.unsqueeze(1)                          # (B,1,H,W,T)
        else:
            eeg = eeg.permute(0, 4, 1, 2, 3)                # (B,H,W,T,1) -> (B,1,H,W,T)
        B, S = fnirs.shape[:2]
        fnirs = fnirs.permute(0, 1, 5, 2, 3, 4)             # (B,S,2,H,W,T)

        fu1, e1, f1, pcc1 = self.b1(eeg, eeg, fnirs)
        fu1, e1, f1 = self.drop(fu1), self.drop(e1), self.drop(f1)
        fu2, e2, f2, pcc2 = self.b2(fu1, e1, f1)
        fu2, e2, f2 = (self.drop(x.mean(dim=-1, keepdim=True)) for x in (fu2, e2, f2))

        fu_flat = self._flat_cl(fu2)                                                  # (B,512)
        fn_flat = torch.stack([self._flat_cl(f2[:, s]) for s in range(S)], dim=1)     # (B,S,512)
        fusion_feat, fnirs_feat, att, pcc_ef = self.egta(fu_flat, fn_flat)

        fu_h = F.elu(self.fu_fc(F.elu(fusion_feat)))
        fn_h = F.elu(self.fn_fc(F.elu(fnirs_feat)))
        eeg_h = F.elu(self.eeg_fc(self._flat_cl(e2)))
        p_fu = torch.softmax(self.fu_out(fu_h), dim=-1)
        p_fn = torch.softmax(self.fn_out(fn_h), dim=-1)
        p_eeg = torch.softmax(self.eeg_out(eeg_h), dim=-1)
        w = torch.softmax(torch.cat([self.fu_w(fu_h), self.fn_w(fn_h)], dim=-1), dim=-1)   # (B,2)
        p_final = w[:, 0:1] * p_fu + w[:, 1:2] * p_fn
        reg = (1 - pcc1) + (1 - pcc2) + (1 - pcc_ef)
        return {"p_final": p_final, "p_eeg": p_eeg, "p_fusion": p_fu, "p_fnirs": p_fn,
                "w": w, "att": att, "reg": reg}


def sta_loss(out, y_onehot, eps=1e-7):
    """Keras categorical_crossentropy on probabilities (clipped) for class_output and eeg_output,
    plus the add_loss PCC regularisers. Returns total, class_ce."""
    ce = lambda p: -(y_onehot * torch.log(p.clamp(eps, 1 - eps))).sum(dim=1).mean()
    c_cls, c_eeg = ce(out["p_final"]), ce(out["p_eeg"])
    return c_cls + c_eeg + out["reg"], c_cls
