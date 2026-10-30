"""Run TitaNet over a long utterance with ONE fixed-shape window, and still get the full-utterance embedding.

The memory problem this exists to solve: in a Core AI `.aimodel`, the activation arena is sized per compiled
shape and allocated for *every* function in the bundle, so a bucket ladder costs the sum of its shapes
(measured: 7 buckets = 768-1199 MB; one f1024 function = 127 MB).  One small window is the cheapest possible
bundle -- but TitaNet has three global reductions over time, so a naive window is not the same function:

  * five SqueezeExcite gates, each sigmoid(W2 relu(W1 mean_t(block_j activation))), se_context_size = -1;
  * the pooling mean/std of the encoder output;
  * the attention softmax over time.

All three are reconstructed here from per-window partial sums, so the windowed run reproduces the whole-utterance
arithmetic rather than approximating it.

Convolution locality.  Every conv is stride 1 with "same" padding; the stack's receptive field is 93 frames, so a
window's interior is exact once it carries MARGIN = 46 frames of *real* context on each side.  `windows()` lays the
kept interiors end to end with no gap and no overlap, and the first/last windows inherit the utterance's own
zero-padding, exactly as a whole-utterance run would see it.

The gates are a triangular fixed point: block j's mean depends only on gates < j, so iterating
(gates -> means -> gates) is exact after 5 sweeps and in practice settles far sooner.  `se_mode="global"` does
that; `se_mode="local"` is the cheap control that lets each window use its own mean, which is what a naive
chunked port would do.

Nothing here stores the encoder output for the whole utterance: the pooling statistics come from running sums and
the attention from an online (running-max) softmax, so host memory is O(channels), not O(channels x frames).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import model as M


def windows(n: int, F: int, margin: int = M.MARGIN) -> list[tuple[int, int, int]]:
    """[(start, keep_lo, keep_hi)] in global frame coordinates: interiors tile [0, n) exactly once."""
    if n <= F:
        return [(0, 0, n)]
    step = F - 2 * margin
    if step <= 0:
        raise ValueError(f"window {F} is too small for a {margin}-frame margin on each side")
    out, s = [], 0
    while True:
        end = s + F
        lo = 0 if s == 0 else s + margin
        hi = n if end >= n else end - margin
        out.append((s, lo, hi))
        if end >= n:
            return out
        s += step


def _window_inputs(mel: torch.Tensor, n: int, s: int, F: int, keep: tuple[int, int]):
    """mel [1, 80, >=n] -> (mel_w [1, 80, F], valid_w [1, 1, F], keep_w [1, 1, F])."""
    dt, dv = mel.dtype, mel.device
    mel_w = torch.zeros(1, M.N_MELS, F, dtype=dt, device=dv)
    take = max(0, min(F, n - s))
    if take:
        mel_w[:, :, :take] = mel[:, :, s:s + take]
    valid_w = torch.zeros(1, 1, F, dtype=dt, device=dv)
    valid_w[0, 0, :take] = 1
    keep_w = torch.zeros(1, 1, F, dtype=dt, device=dv)
    keep_w[0, 0, keep[0] - s:keep[1] - s] = 1
    return mel_w, valid_w, keep_w


@dataclass
class StreamResult:
    emb: torch.Tensor
    sweeps: int
    gate_delta: float
    n_windows: int
    evaluations: int


@torch.no_grad()
def embed(model: M.TitaNet, mel: torch.Tensor, n: int, F: int, *, se_mode: str = "global",
          max_sweeps: int = 5, tol: float = 1e-6, margin: int = M.MARGIN) -> StreamResult:
    """The whole-utterance embedding, computed with one fixed window of F frames.

    mel: [1, 80, >= n] with the utterance's n valid frames first.  Returns emb [1, 192].
    """
    wins = windows(n, F, margin)
    dt, dv = mel.dtype, mel.device
    nt = torch.tensor([[float(n)]], dtype=dt, device=dv)
    evals = 0

    # ---- 1. the five SqueezeExcite gates ------------------------------------------------------
    gates = None
    sweeps, delta = 0, float("nan")
    if se_mode == "global":
        for sweep in range(max_sweeps):
            sums = {j: torch.zeros(1, M.SE_CHANNELS[j], dtype=dt, device=dv) for j in range(5)}
            for s, lo, hi in wins:
                mw, vw, kw = _window_inputs(mel, n, s, F, (lo, hi))
                col = M.Collected()
                model.enc(mw, vw, nt, gates=gates, collect=col, keep=kw)
                evals += 1
                for j in range(5):
                    sums[j] = sums[j] + col.se[j]
            new = [model.enc.blocks[j].se.gate(sums[j] / nt) for j in range(5)]
            sweeps = sweep + 1
            delta = 0.0 if gates is None else max(float((a - b).abs().max()) for a, b in zip(new, gates))
            gates = new
            if gates is not None and sweep > 0 and delta < tol:
                break
    elif se_mode != "local":
        raise ValueError(se_mode)

    # ---- 2. the pooling mean / std of the encoder output --------------------------------------
    sx = torch.zeros(1, M.C_OUT, dtype=dt, device=dv)
    sxx = torch.zeros(1, M.C_OUT, dtype=dt, device=dv)
    for s, lo, hi in wins:
        mw, vw, kw = _window_inputs(mel, n, s, F, (lo, hi))
        x = model.enc(mw, vw, nt, gates=gates)
        evals += 1
        xk = x * kw
        sx = sx + xk.sum(dim=2)
        sxx = sxx + (xk * x).sum(dim=2)
    mean, std = M.AttentivePool.stats_from_sums(sx, sxx, nt)

    # ---- 3. the attention softmax over time, online -------------------------------------------
    neg = torch.finfo(dt if dt.is_floating_point else torch.float32).min
    mx = torch.full((1, M.C_OUT), neg, dtype=dt, device=dv)
    den = torch.zeros(1, M.C_OUT, dtype=dt, device=dv)
    ax = torch.zeros(1, M.C_OUT, dtype=dt, device=dv)
    axx = torch.zeros(1, M.C_OUT, dtype=dt, device=dv)
    for s, lo, hi in wins:
        mw, vw, kw = _window_inputs(mel, n, s, F, (lo, hi))
        x = model.enc(mw, vw, nt, gates=gates)
        evals += 1
        lg = model.pool.logits(x, mean, std)
        lg = torch.where(kw.expand_as(lg) > 0, lg, torch.full_like(lg, neg))
        m_new = torch.maximum(mx, lg.max(dim=2).values)
        scale = torch.exp(mx - m_new)
        w = torch.exp(lg - m_new.unsqueeze(2)) * kw
        den = den * scale + w.sum(dim=2)
        ax = ax * scale + (w * x).sum(dim=2)
        axx = axx * scale + (w * x * x).sum(dim=2)
        mx = m_new
    mu = ax / den
    sg = torch.sqrt((axx / den - mu * mu).clamp(min=M.STD_EPS))

    emb = model.head(torch.cat([mu, sg], dim=1))
    return StreamResult(emb=emb, sweeps=sweeps, gate_delta=delta, n_windows=len(wins), evaluations=evals)


@torch.no_grad()
def embed_segment_average(model: M.TitaNet, mel: torch.Tensor, n: int, F: int) -> torch.Tensor:
    """The cheap control: independent embeddings of F-frame segments, averaged after L2 normalisation.

    Standard practice for long-form speaker embedding, and the thing to beat: it needs no aggregation at all,
    but each segment sees only its own global statistics, so it is not the whole-utterance function.
    """
    dt = mel.dtype
    outs = []
    s = 0
    while s < n:
        take = min(F, n - s)
        if take < F and s > 0:           # right-align the tail so every segment is a full window of real audio
            s = max(0, n - F)
            take = min(F, n - s)
        mw, vw, _ = _window_inputs(mel, n, s, F, (s, s + take))
        e = model(mw, vw)
        outs.append(e / e.norm(dim=1, keepdim=True))
        s += F
    return torch.stack(outs).mean(dim=0)
