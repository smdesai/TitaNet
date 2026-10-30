"""TitaNet-Large as a fixed-shape torch graph, authored for *chunked* execution and a small activation arena.

Read straight from the publisher's checkpoint (`model_weights.ckpt`); no NeMo import. Every op mirrors
NVIDIA-NeMo/Speech 3.1.0 in eval mode:

  encoder  ConvASREncoder, 5 JasperBlocks, stride 1, "same" padding, every conv a MaskedConv1d (its input's
           padded frames zeroed first):
             block 0  prolog  80 -> 1024   k3   1 sub-block   no residual
             block 1  mega    1024 -> 1024 k7   3 sub-blocks  residual (1x1 conv -> BN)
             block 2  mega    1024 -> 1024 k11  3 sub-blocks  residual
             block 3  mega    1024 -> 1024 k15  3 sub-blocks  residual
             block 4  epilog  1024 -> 3072 k1   1 sub-block   no residual
           sub-block = depthwise conv -> pointwise conv -> BatchNorm(eps 1e-3) [-> ReLU between sub-blocks];
           after the last sub-block: SqueezeExcite (se_context_size -1, i.e. GLOBAL over valid frames),
           then + residual, then ReLU.
  decoder  SpeakerDecoder / AttentivePoolLayer: global masked mean/std of x -> cat(x, mean, std) [9216] ->
           Conv1d(9216 -> 128) -> tanh -> Conv1d(128 -> 3072) -> -inf at pads -> softmax over time ->
           attention-weighted mean/std -> cat [6144] -> BatchNorm(eps 1e-5) -> Conv1d(6144 -> 192) = emb.

Two authoring decisions that this module exists to make possible, both aimed at runtime memory:

1. `fold_pool_concat` (default on).  NeMo materialises cat(x, mean, std), a [1, 9216, T] tensor, only to run a
   1x1 conv over it.  Since mean/std are constant along time, that conv is
       W_x (*) x  +  (W_mean @ mean + W_std @ std + b)
   i.e. a 3072->128 conv plus one broadcast 128-vector.  Identical arithmetic, no 9216-channel tensor.

2. The three global reductions (the five SqueezeExcite means, the pooling mean/std, the attention softmax) are
   *injectable* and *reportable*.  `gates=`/`stats=` supply them from outside; `collect=` reports the partial
   sums a caller needs to build them.  That is what lets a host run one small fixed window repeatedly over a long
   utterance and still reproduce the full-utterance embedding exactly, instead of compiling a function per length.

Receptive field: 1 + 2 + 3*6 + 3*10 + 3*14 + 0 = 93 frames, so a chunk needs MARGIN = 46 frames of real context
on each side for its interior to equal the full-utterance result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS, C, C_OUT, EMB, ATT = 80, 1024, 3072, 192, 128
KERNELS = (3, 7, 11, 15, 1)
REPEATS = (1, 3, 3, 3, 1)
RESIDUAL = (False, True, True, True, False)
SE_CHANNELS = (C, C, C, C, C_OUT)
ENC_BN_EPS, DEC_BN_EPS = 1e-3, 1e-5
STD_EPS = 1e-10
HOP = 160
RECEPTIVE_FIELD = 1 + 2 + 3 * 6 + 3 * 10 + 3 * 14
MARGIN = RECEPTIVE_FIELD // 2          # 46


def load_state_dict(snapshot: Path) -> dict:
    return torch.load(Path(snapshot) / "nemo" / "model_weights.ckpt", map_location="cpu", weights_only=True)


def mel_frames(n_samples: int) -> tuple[int, int]:
    """(frames, valid) of NeMo's preprocessor: N // 160 + 1 frames, of which N // 160 carry signal."""
    return n_samples // HOP + 1, n_samples // HOP


def cast(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == x.dtype else t.to(x.dtype)


@dataclass
class Collected:
    """The partial sums a chunk contributes to the model's global reductions.

    Every entry is a plain sum over the frames the caller marked with `keep`, so a host adds them across chunks
    and divides once.  `se[j]` is the sum over time of block j's SqueezeExcite input (its pre-gate activation).
    """
    se: dict[int, torch.Tensor] = field(default_factory=dict)
    x_sum: torch.Tensor | None = None            # sum_t x            [C_OUT]
    x_sqsum: torch.Tensor | None = None          # sum_t x^2          [C_OUT]
    logits: torch.Tensor | None = None           # attention logits   [C_OUT, T]
    x: torch.Tensor | None = None                # encoder output     [C_OUT, T]


class BN(nn.Module):
    """Eval-mode BatchNorm1d over the channels of [B, C, T], from the running statistics."""

    def __init__(self, n: int, eps: float):
        super().__init__()
        self.eps = eps
        for k in ("weight", "bias", "running_mean", "running_var"):
            self.register_buffer(k, torch.zeros(n))

    def forward(self, x):
        return F.batch_norm(x, cast(self.running_mean, x), cast(self.running_var, x), cast(self.weight, x),
                            cast(self.bias, x), False, 0.0, self.eps)

    def load(self, sd: dict, prefix: str) -> "BN":
        for k in ("weight", "bias", "running_mean", "running_var"):
            setattr(self, k, sd[f"{prefix}.{k}"].clone())
        return self


class SubBlock(nn.Module):
    """MaskedConv1d depthwise (k, groups=C_in) -> MaskedConv1d pointwise (1x1) -> BatchNorm."""

    def __init__(self, c_in: int, c_out: int, k: int):
        super().__init__()
        self.k, self.c_in = k, c_in
        self.register_buffer("dw", torch.zeros(c_in, 1, k))
        self.register_buffer("pw", torch.zeros(c_out, c_in, 1))
        self.bn = BN(c_out, ENC_BN_EPS)

    def forward(self, x, m):
        x = F.conv1d(x * m, cast(self.dw, x), padding=self.k // 2, groups=self.c_in)
        x = F.conv1d(x * m, cast(self.pw, x))
        return self.bn(x)

    def load(self, sd: dict, prefix: str, i: int) -> "SubBlock":
        self.dw = sd[f"{prefix}.mconv.{i}.conv.weight"].clone()
        self.pw = sd[f"{prefix}.mconv.{i + 1}.conv.weight"].clone()
        self.bn.load(sd, f"{prefix}.mconv.{i + 2}")
        return self


class SqueezeExcite(nn.Module):
    """Global squeeze-excite: sigmoid(fc2 ReLU(fc1 mean_t(x))) as a per-channel gate, no biases.

    `stat_scale` rescales the mean as sum(x * (s / n)) / s.  Exact in real arithmetic; it exists because in fp16
    NeMo's own form multiplies weights of order 1/n by activations of order 1e-3 and the products land in the
    subnormal range.
    """

    def __init__(self, c: int):
        super().__init__()
        self.register_buffer("fc1", torch.zeros(c // 8, c))
        self.register_buffer("fc2", torch.zeros(c, c // 8))
        self.stat_scale: float | None = None

    def mean(self, x, m, n):
        """Masked global mean over the n valid frames of [B, C, T]."""
        xm = x * m
        if self.stat_scale is None:
            return xm.sum(dim=2) / n
        s = self.stat_scale
        return (xm * (s / n)).sum(dim=2) / s

    def gate(self, mu):
        """mu [B, C] -> gate [B, C, 1]."""
        h = F.relu(F.linear(mu, cast(self.fc1, mu)))
        return torch.sigmoid(F.linear(h, cast(self.fc2, h))).unsqueeze(2)

    def forward(self, x, m, n, gate=None):
        xm = x * m
        g = self.gate(self.mean(x, m, n)) if gate is None else cast(gate, x)
        return xm * g

    def load(self, sd: dict, prefix: str, i: int) -> "SqueezeExcite":
        self.fc1 = sd[f"{prefix}.mconv.{i}.fc.0.weight"].clone()
        self.fc2 = sd[f"{prefix}.mconv.{i}.fc.2.weight"].clone()
        return self


class Block(nn.Module):
    """A JasperBlock: `repeat` sub-blocks, SqueezeExcite on the last, optional residual, then ReLU."""

    def __init__(self, idx: int, c_in: int, c_out: int, k: int, repeat: int, residual: bool):
        super().__init__()
        self.idx = idx
        self.subs = nn.ModuleList([SubBlock(c_in if i == 0 else c_out, c_out, k) for i in range(repeat)])
        self.se = SqueezeExcite(c_out)
        self.has_res = residual
        if residual:
            self.register_buffer("res", torch.zeros(c_out, c_in, 1))
            self.res_bn = BN(c_out, ENC_BN_EPS)

    def forward(self, x, m, n, gate=None, collect: Collected | None = None, keep=None):
        inp = x
        for i, sub in enumerate(self.subs):
            x = sub(x, m)
            if i + 1 < len(self.subs):
                x = F.relu(x)
        if collect is not None:
            # the sum over the kept frames of this block's SqueezeExcite input
            collect.se[self.idx] = (x * (keep if keep is not None else m)).sum(dim=2)
        x = self.se(x, m, n, gate)
        if self.has_res:
            r = F.conv1d(inp * m, cast(self.res, inp))
            x = x + self.res_bn(r)
        return F.relu(x)

    def load(self, sd: dict, prefix: str) -> "Block":
        # mconv indices: each sub-block occupies (dw, pw, bn) then (relu, dropout) between sub-blocks
        for i, sub in enumerate(self.subs):
            sub.load(sd, prefix, i * 5)
        self.se.load(sd, prefix, len(self.subs) * 5 - 2)
        if self.has_res:
            self.res = sd[f"{prefix}.res.0.0.conv.weight"].clone()
            self.res_bn.load(sd, f"{prefix}.res.0.1")
        return self


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        c_in = [N_MELS, C, C, C, C]
        c_out = [C, C, C, C, C_OUT]
        self.blocks = nn.ModuleList([Block(i, c_in[i], c_out[i], KERNELS[i], REPEATS[i], RESIDUAL[i])
                                     for i in range(5)])

    def forward(self, x, m, n, gates=None, collect: Collected | None = None, keep=None, depth: int = 5):
        """gates: optional list of 5 per-channel gates [B, C, 1] (None entries are computed locally).

        `depth` runs only the first `depth` blocks, which is what a progressive gate scan needs.
        """
        for j, blk in enumerate(self.blocks[:depth]):
            g = None if gates is None else gates[j]
            x = blk(x, m, n, g, collect, keep)
        return x

    def load(self, sd: dict) -> "Encoder":
        for i, blk in enumerate(self.blocks):
            blk.load(sd, f"encoder.encoder.{i}")
        return self


class AttentivePool(nn.Module):
    """AttentivePoolLayer, with the 9216-channel concat optionally folded into the first conv."""

    def __init__(self, fold: bool = True):
        super().__init__()
        self.fold = fold
        self.register_buffer("att0_w", torch.zeros(ATT, 3 * C_OUT, 1))
        self.register_buffer("att0_b", torch.zeros(ATT))
        self.bn = BN(ATT, DEC_BN_EPS)
        self.register_buffer("att2_w", torch.zeros(C_OUT, ATT, 1))
        self.register_buffer("att2_b", torch.zeros(C_OUT))
        self.stat_scale: float | None = None

    # ---- global statistics of the encoder output -------------------------------------------------
    def stats(self, x, m, n):
        """NeMo's get_statistics_with_mask(x, mask / n): mean and std over the valid frames."""
        xm = x * m
        if self.stat_scale is None:
            mean = xm.sum(dim=2) / n
        else:
            s = self.stat_scale
            mean = (xm * (s / n)).sum(dim=2) / s
        d = (x - mean.unsqueeze(2)) * m
        if self.stat_scale is None:
            var = (d * d).sum(dim=2) / n
        else:
            s = self.stat_scale
            var = ((d * d) * (s / n)).sum(dim=2) / s
        return mean, torch.sqrt(var.clamp(min=STD_EPS))

    @staticmethod
    def stats_from_sums(x_sum, x_sqsum, n):
        """The same mean/std from plain sums, for a host that accumulated them chunk by chunk."""
        mean = x_sum / n
        var = (x_sqsum / n) - mean * mean
        return mean, torch.sqrt(var.clamp(min=STD_EPS))

    # ---- attention logits -----------------------------------------------------------------------
    def logits(self, x, mean, std):
        """TDNNModule(9216 -> 128) = bn(relu(conv)) over cat(x, mean, std), then tanh, then Conv1d(128 -> 3072)."""
        if self.fold:
            w = cast(self.att0_w, x)
            wx, wm, ws = w[:, :C_OUT], w[:, C_OUT:2 * C_OUT], w[:, 2 * C_OUT:]
            h = F.conv1d(x, wx, cast(self.att0_b, x))
            col = (F.linear(mean, wm[:, :, 0]) + F.linear(std, ws[:, :, 0])).unsqueeze(2)
            h = h + col
        else:
            T = x.shape[2]
            attn = torch.cat([x, mean.unsqueeze(2).expand(-1, -1, T), std.unsqueeze(2).expand(-1, -1, T)], dim=1)
            h = F.conv1d(attn, cast(self.att0_w, x), cast(self.att0_b, x))
        h = torch.tanh(self.bn(F.relu(h)))      # TDNNModule: conv -> ReLU -> BatchNorm, then the Sequential's Tanh
        return F.conv1d(h, cast(self.att2_w, h), cast(self.att2_b, h))

    def weighted_stats(self, x, a):
        """NeMo's get_statistics_with_mask(x, alpha): the attention-weighted mean and std.

        `stat_scale` folds a power of two into the weights before multiplying.  Exact in real arithmetic, and
        required in fp16: alpha is of order 1/T (~1e-3) and (x - mu)^2 of order 1e-2, so the unscaled product
        alpha * (x - mu)^2 is ~1e-5 -- below fp16's smallest normal, 6.1e-5.
        """
        s = self.stat_scale
        aw = a if s is None else a * s
        mu = (x * aw).sum(dim=2)
        if s is not None:
            mu = mu / s
        d = x - mu.unsqueeze(2)
        var = ((d * d) * aw).sum(dim=2)
        if s is not None:
            var = var / s
        return mu, torch.sqrt(var.clamp(min=STD_EPS))

    def forward(self, x, m, n):
        mean, std = self.stats(x, m, n)
        lg = self.logits(x, mean, std)
        lg = lg + torch.log(m)                      # -inf at the padded frames, in-graph
        a = torch.softmax(lg, dim=2)
        mu, sg = self.weighted_stats(x, a)
        return torch.cat([mu, sg], dim=1)

    def load(self, sd: dict) -> "AttentivePool":
        p = "decoder._pooling.attention_layer"
        self.att0_w = sd[f"{p}.0.conv_layer.weight"].clone()
        self.att0_b = sd[f"{p}.0.conv_layer.bias"].clone()
        self.bn.load(sd, f"{p}.0.bn")
        self.att2_w = sd[f"{p}.2.weight"].clone()
        self.att2_b = sd[f"{p}.2.bias"].clone()
        return self


class TitaNet(nn.Module):
    """mel [B, 80, T] (+ mask) -> emb [B, 192]."""

    def __init__(self, fold_pool_concat: bool = True):
        super().__init__()
        self.enc = Encoder()
        self.pool = AttentivePool(fold=fold_pool_concat)
        self.emb_bn = BN(2 * C_OUT, DEC_BN_EPS)
        self.register_buffer("emb_w", torch.zeros(EMB, 2 * C_OUT, 1))
        self.register_buffer("emb_b", torch.zeros(EMB))

    def head(self, pooled):
        e = self.emb_bn(pooled.unsqueeze(2))[:, :, 0]
        return F.linear(e, cast(self.emb_w, e)[:, :, 0], cast(self.emb_b, e))

    def forward(self, mel, valid):
        n = valid.sum(dim=2)                       # [B, 1]
        x = self.enc(mel, valid, n)
        return self.head(self.pool(x, valid, n))

    def load(self, sd: dict) -> "TitaNet":
        self.enc.load(sd)
        self.pool.load(sd)
        self.emb_bn.load(sd, "decoder.emb_layers.0.0")
        self.emb_w = sd["decoder.emb_layers.0.1.weight"].clone()
        self.emb_b = sd["decoder.emb_layers.0.1.bias"].clone()
        return self

    @classmethod
    def from_snapshot(cls, snapshot: Path, sd: dict | None = None, fold_pool_concat: bool = True) -> "TitaNet":
        m = cls(fold_pool_concat=fold_pool_concat)
        m.load(sd if sd is not None else load_state_dict(snapshot))
        return m.eval()
