"""Precision recipes for the exported graph.

  fp32   the checkpoint's own arithmetic; the control
  wfp16  fp16 weight storage, fp32 compute (coreai-torch keeps an fp16 constant fp16 in the bundle, and the cast
         at use is free); BatchNorm statistics stay fp32.  Halves the weight bytes without touching activations.
  fp16   fp16 weights and activations throughout, with the three global reductions rescaled by a power of two.
         The rescale is exact in real arithmetic and necessary in fp16: the SqueezeExcite and pooling means
         multiply weights of order 1/n by activations of order 1e-3, and the products land in fp16 subnormals.

`fp16` is the recipe to ship: it halves both the weight residency and the activation arena, which is what the
memory budget is actually spent on.
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import model as T

ARMS = ("fp32", "wfp16", "fp16")
WEIGHT_BUFFERS = ("dw", "pw", "res", "fc1", "fc2", "att0_w", "att2_w", "emb_w")
STAT_SCALE = 256.0


def compute_dtype(arm: str) -> torch.dtype:
    return torch.float32 if arm in ("fp32", "wfp16") else torch.float16


def output_dtype(arm: str) -> torch.dtype:
    return torch.float32 if arm in ("fp32", "wfp16") else torch.float16


def build(arm: str, snapshot: Path, sd: dict | None = None, fold_pool_concat: bool = True) -> T.TitaNet:
    assert arm in ARMS, arm
    m = T.TitaNet.from_snapshot(snapshot, sd, fold_pool_concat=fold_pool_concat)
    if arm == "wfp16":
        for mod in m.modules():
            for name, buf in list(mod.named_buffers(recurse=False)):
                if name in WEIGHT_BUFFERS:
                    setattr(mod, name, buf.to(torch.float16))
    elif arm == "fp16":
        m.half()
        for b in m.enc.blocks:
            b.se.stat_scale = STAT_SCALE
        m.pool.stat_scale = STAT_SCALE
    return m.eval()
