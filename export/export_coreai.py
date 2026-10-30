"""Export a Core AI JIT IR bundle (`titanet.aimodel`) for ONE fixed window.

  python export/export_coreai.py --arm fp16 --shape 1024 --fn once --out DIR
  python export/export_coreai.py --arm fp16 --shape 512  --fn scan --out DIR

Why one function.  Measured on an iPhone19,4 (`phys_footprint`, GPU, fp16): a 7-bucket bundle costs 768 MB even
when only one bucket is ever called, because the activation arena is allocated per compiled shape for every
function in the bundle; one f1024 function costs 127 MB, and 98 MB once the pooling concat is folded out.  So the
bundle here has exactly one entrypoint and the host reaches long audio by calling it repeatedly, not by compiling
more shapes.

Entrypoints
  once   mel [1, 80, F] . valid [1, 1, F]                       -> emb [1, 192]
         the whole model; exact for an utterance of at most F frames.  A latency fast path, nothing more.
  scan   mel . valid . keep [1, 1, F] . n_inv . s0..s4 . mean . std
                                       -> se0..se4, x_sum, x_sqsum, lmax, den, ax, axx
         one window's contribution to all three global reductions.  `keep` selects the window's interior so the
         contributions tile the utterance exactly once; `s0..s4` are the previous sweep's *summed* SqueezeExcite
         means, so the gate MLPs stay inside the graph and the host needs no model weights at all.  Passing
         s_j = 0 on the first sweep starts the fixed point at sigmoid(0).
  head   mu [1, 3072] . sg [1, 3072]                            -> emb [1, 192]
         the embedding head, exported beside `scan` because it has no time axis and so costs no arena.

The host between them (tnl.stream / tnl.runtime) only ever adds, divides, exponentiates and compares 3072-wide
vectors, so its own memory is O(channels) rather than O(channels x frames).

Before saving, every exported program is replayed in torch against the eager module and the drift is recorded; a
drift above 1e-3 fails the export rather than shipping a graph that is not the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import arms as A  # noqa: E402
from tnl import model as M  # noqa: E402
from tnl import paths as P  # noqa: E402

SNAP = P.SNAPSHOT

ONCE_IN = ["mel", "valid"]
ONCE_OUT = ["emb"]
SCAN_IN = ["mel", "valid", "keep", "n_inv", "s0", "s1", "s2", "s3", "s4", "mean", "std"]
SCAN_OUT = ["se0", "se1", "se2", "se3", "se4", "x_sum", "x_sqsum", "lmax", "den", "ax", "axx"]
HEAD_IN = ["mu", "sg"]
HEAD_OUT = ["emb"]


class Once(torch.nn.Module):
    def __init__(self, net: M.TitaNet):
        super().__init__()
        self.net = net

    def forward(self, mel, valid):
        return self.net(mel, valid)


class Head(torch.nn.Module):
    """cat(mu, sigma) -> BatchNorm -> Conv1d(6144 -> 192).  No time axis."""

    def __init__(self, net: M.TitaNet):
        super().__init__()
        self.net = net

    def forward(self, mu, sg):
        return self.net.head(torch.cat([mu, sg], dim=1))


class Scan(torch.nn.Module):
    """One window's contribution to the five SqueezeExcite means, the pooling statistics and the attention.

    The five gates are rebuilt in-graph from the previous sweep's summed means (s0..s4) and 1/n, so a host can
    drive the fixed point without holding any of the model's weights.
    """

    def __init__(self, net: M.TitaNet):
        super().__init__()
        self.net = net

    def forward(self, mel, valid, keep, n_inv, s0, s1, s2, s3, s4, mean, std):
        net = self.net
        n = 1.0 / n_inv
        gates = [net.enc.blocks[j].se.gate(s * n_inv) for j, s in enumerate((s0, s1, s2, s3, s4))]
        col = M.Collected()
        x = net.enc(mel, valid, n, gates=gates, collect=col, keep=keep)
        xk = x * keep
        x_sum = xk.sum(dim=2)
        x_sqsum = (xk * x).sum(dim=2)
        lg = net.pool.logits(x, mean, std)
        neg = torch.finfo(lg.dtype).min
        lg = torch.where(keep.expand_as(lg) > 0, lg, torch.full_like(lg, neg))
        lmax = lg.max(dim=2).values
        w = torch.exp(lg - lmax.unsqueeze(2)) * keep
        den = w.sum(dim=2)
        ax = (w * x).sum(dim=2)
        axx = (w * x * x).sum(dim=2)
        return (col.se[0], col.se[1], col.se[2], col.se[3], col.se[4],
                x_sum, x_sqsum, lmax, den, ax, axx)


def sample_inputs(fn: str, F: int, dtype) -> tuple:
    g = torch.Generator().manual_seed(7)
    mel = torch.randn(1, M.N_MELS, F, generator=g).to(dtype)
    valid = torch.ones(1, 1, F, dtype=dtype)
    if fn == "once":
        return (mel, valid)
    if fn == "head":
        return (torch.randn(1, M.C_OUT, generator=g).to(dtype) * 0.1,
                (torch.rand(1, M.C_OUT, generator=g).to(dtype) * 0.1 + 0.01))
    keep = torch.ones(1, 1, F, dtype=dtype)
    n_inv = torch.full((1, 1), 1.0 / F, dtype=dtype)
    sums = [torch.randn(1, M.SE_CHANNELS[j], generator=g).to(dtype) * float(F) * 0.01 for j in range(5)]
    mean = torch.randn(1, M.C_OUT, generator=g).to(dtype) * 0.1
    std = (torch.rand(1, M.C_OUT, generator=g).to(dtype) * 0.1 + 0.01)
    return (mel, valid, keep, n_inv, *sums, mean, std)


def flat(out):
    return [out] if isinstance(out, torch.Tensor) else list(out)


def drift(a, b) -> float:
    d = 0.0
    for x, y in zip(flat(a), flat(b)):
        x, y = x.double(), y.double()
        s = float(y.abs().max()) or 1.0
        d = max(d, float((x - y).abs().max()) / s)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=A.ARMS, default="fp16")
    ap.add_argument("--shape", type=int, required=True, help="window length F in mel frames")
    ap.add_argument("--fn", choices=("once", "scan", "once+scan"), default="once")
    ap.add_argument("--no-fold", action="store_true", help="keep NeMo's 9216-channel pooling concat")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        sys.exit(f"{args.out} exists: an export never overwrites a bundle")

    from coreai_torch import TorchConverter, get_decomp_table
    from coreai_torch.debugging.debug_info import strip_debug_info

    t0 = time.monotonic()
    dtype = A.compute_dtype(args.arm)
    net = A.build(args.arm, SNAP, fold_pool_concat=not args.no_fold)

    kinds = {"once": (Once, ONCE_IN, ONCE_OUT), "scan": (Scan, SCAN_IN, SCAN_OUT),
             "head": (Head, HEAD_IN, HEAD_OUT)}
    # `scan` cannot produce an embedding on its own, so the (time-axis-free, arena-free) head ships with it
    wanted = args.fn.split("+")
    if "scan" in wanted:
        wanted.append("head")

    conv = TorchConverter()
    drifts = {}
    for name in wanted:
        cls, in_names, out_names = kinds[name]
        mod = cls(net).eval()
        inputs = sample_inputs(name, args.shape, dtype)
        with torch.no_grad():
            eager = mod(*inputs)
            ep = torch.export.export(mod, inputs)
            replay = ep.module()(*inputs)
        d = drift(replay, eager)
        drifts[name] = d
        if d > 1e-3:
            sys.exit(f"torch.export changed {name}'s output by {d:.2e}")
        conv.add_exported_program(ep.run_decompositions(get_decomp_table()), input_names=in_names,
                                  output_names=out_names, entrypoint_name=name)
    prog = conv.to_coreai()
    strip_debug_info(prog)
    args.out.mkdir(parents=True)
    aimodel = args.out / "titanet.aimodel"
    prog.save_asset(aimodel)
    size = sum(f.stat().st_size for f in aimodel.rglob("*") if f.is_file())

    rec = {"aimodel": "titanet.aimodel", "arm": args.arm, "entrypoints": wanted, "shape": args.shape,
           "seconds_of_audio": args.shape * M.HOP / 16000, "fold_pool_concat": not args.no_fold,
           "compute_dtype": str(dtype).split(".")[-1], "output_dtype": str(A.output_dtype(args.arm)).split(".")[-1],
           "signature": {n: {"inputs": kinds[n][1], "outputs": kinds[n][2]} for n in wanted},
           "margin": M.MARGIN,
           "receptive_field": M.RECEPTIVE_FIELD, "export_drift": drifts, "bytes": size,
           "export_seconds": round(time.monotonic() - t0, 1),
           "snapshot": json.loads((SNAP / "PROVENANCE.json").read_text())["revision"]}
    (args.out / "export.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
