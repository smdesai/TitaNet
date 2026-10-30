"""Export the same one-window `scan` + `head` pair as Core ML ML Programs (`.mlpackage`), for iOS 17+ / macOS 14+.

  python export/export_coreml.py --precision fp32  --shape 512 --out DIR    # CPU / GPU, fp32 compute
  python export/export_coreml.py --precision fp16  --shape 512 --out DIR    # Neural Engine, fp16 compute
  python export/export_coreml.py --precision fp16c --shape 512 --out DIR    # Neural Engine, fp16, centred attention

The graphs are export/export_coreai.py's `Scan` and `Head` over tnl/model.py, traced and converted with coremltools; the
host loop (tnl/runtime.py, python/titanet_coreai.py on the Hub) is unchanged for fp32 / fp16. Precision:

  fp32   the checkpoint's weights and arithmetic (compute_precision FLOAT32). Core ML runs it on CPU or GPU only.
  fp16   compute_precision FLOAT16 throughout. Runs entirely on the ANE. Worst cosine to NeMo 0.985 (40 clips):
         the attention variance is formed on the host as axx/den - (ax/den)^2 from fp16 partial sums, and that
         cancellation amplifies their ~1e-3 rounding (attribution: taking lmax/den/ax/axx from fp32 alone -> 0.9995;
         the fp16 head costs the last ~4e-4). tnl/arms' stat_scale does not reach these sums: `scan` reports plain sums.
  fp16c  fp16 `scan` with three exact rewrites, and the head in fp32 (one call per clip, CPU):
         1 attention moments centred on the pooling mean the host already feeds it, and scaled by a power of two S:
           ax_c = sum w S (x - mean), axx_c = sum w (S (x - mean))^2; with q = ax_c / (S den), mu = mean + q and
           var = axx_c / (S^2 den) - q^2, so no large terms cancel. The host must send `mean` already rounded to fp16
           and add back that same value (`centred` / `moment_scale` in export.json; verify/g5_coreml_vs_nemo.py).
         2 S = `moment_scale`, default 16: unscaled, 68% of axx's mass is in terms below fp16's smallest normal
           (the encoder output is ~0.02 +- 2.5e-3); at 16 that is 3e-3, with the largest window sum (9e2 over the 40
           fixtures) 73x under fp16's 65504. 64 overflows the random-input replay below. S moved the GPU slightly
           (0.99958 -> 0.99966) and the ANE not at all.
         3 the attention TDNN's BatchNorm gain k = gamma / sqrt(var + eps) folded into the 3072->128 conv before it
           (relu(h) k = relu(k h): every k > 0, checked at export). The conv's outputs are ~0.08 and k reaches 340;
           the ANE (not the GPU) loses ~5% on that conv-then-BatchNorm pair (post-BN error 5e-2, folded 6e-4).
         Worst / mean cosine to NeMo over the 40 fixtures: ANE 0.9968 / 0.9986 (fp16: 0.985 / 0.995), GPU 0.99966.
         What the ANE still loses is the encoder's own fp16 precision: the deviation of its output from the time mean
         (what the attention variance is made of) is 1.9% off on the ANE, 0.2% on the GPU.

Each converted model is replayed through Core ML on this Mac against the eager torch module before it is saved.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import arms as A  # noqa: E402
from tnl import model as M  # noqa: E402
from tnl import paths as P  # noqa: E402

_spec = importlib.util.spec_from_file_location("tnl_export", ROOT / "export" / "export_coreai.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

SNAP = P.SNAPSHOT


SCAN_C_OUT = E.SCAN_OUT[:-2] + ["ax_c", "axx_c"]
MOMENT_SCALE = 16.0         # set by --moment-scale


class ScanCentred(E.Scan):
    """export_coreai.py's Scan with the attention moments centred on `mean` (the pooling mean, constant across windows, so
    the host's running-max combination is unchanged) and scaled by MOMENT_SCALE (a power of two, so exact):
    returns ax_c = sum w S (x - mean), axx_c = sum w (S (x - mean))^2."""

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
        xc = (x - mean.unsqueeze(2)) * MOMENT_SCALE
        ax_c = (w * xc).sum(dim=2)
        axx_c = (w * xc * xc).sum(dim=2)
        return (col.se[0], col.se[1], col.se[2], col.se[3], col.se[4],
                x_sum, x_sqsum, lmax, den, ax_c, axx_c)


def fold_attention_bn(net: M.TitaNet) -> None:
    """relu(h) * k + c == relu(k h) + c for k > 0: move the attention BatchNorm's gain into the conv before it."""
    p, bn = net.pool, net.pool.bn
    k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    if not bool((k > 0).all()):
        sys.exit("attention BatchNorm has a non-positive gain: folding it through the ReLU would not be exact")
    p.att0_w = p.att0_w * k[:, None, None]
    p.att0_b = p.att0_b * k
    bias = bn.bias - k * bn.running_mean
    bn.weight, bn.bias = torch.ones_like(k), bias
    bn.running_mean, bn.running_var = torch.zeros_like(k), torch.full_like(k, 1.0 - bn.eps)   # (x - 0) / 1 exactly


def build(precision: str) -> M.TitaNet:
    net = A.build("fp32", SNAP).eval()
    if precision == "fp16c":
        fold_attention_bn(net)
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=("fp32", "fp16", "fp16c"), required=True)
    ap.add_argument("--shape", type=int, default=512, help="window length F in mel frames")
    ap.add_argument("--target", default="iOS17", help="minimum deployment target (coremltools.target.<name>)")
    ap.add_argument("--moment-scale", type=float, default=16.0, help="fp16c: power-of-two scale of the attention moments")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists: an export never overwrites a bundle")

    import coremltools as ct

    global MOMENT_SCALE
    MOMENT_SCALE = args.moment_scale

    t0 = time.monotonic()
    net = build(args.precision)
    centred = args.precision == "fp16c"
    f32, f16 = ct.precision.FLOAT32, ct.precision.FLOAT16
    kinds = {"scan": (ScanCentred if centred else E.Scan, E.SCAN_IN, SCAN_C_OUT if centred else E.SCAN_OUT,
                      "TitaNetScan", f32 if args.precision == "fp32" else f16),
             "head": (E.Head, E.HEAD_IN, E.HEAD_OUT, "TitaNetHead", f16 if args.precision == "fp16" else f32)}
    args.out.mkdir(parents=True)
    rec_models, drifts = {}, {}
    for name, (cls, in_names, out_names, file, cp) in kinds.items():
        mod = cls(net).eval()
        inputs = E.sample_inputs(name, args.shape, torch.float32)
        with torch.no_grad():
            eager = E.flat(mod(*inputs))
            traced = torch.jit.trace(mod, inputs)
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name=n, shape=tuple(x.shape), dtype=np.float32) for n, x in zip(in_names, inputs)],
            outputs=[ct.TensorType(name=n, dtype=np.float32) for n in out_names],
            convert_to="mlprogram",
            compute_precision=cp,
            minimum_deployment_target=getattr(ct.target, args.target),
        )
        prec = "fp32" if cp == f32 else "fp16"
        ml.short_description = f"TitaNet-Large {name} ({prec}{', centred attention' if centred and name == 'scan' else ''}, " \
            f"{args.shape}-frame window)"
        path = args.out / f"{file}.mlpackage"
        ml.save(str(path))
        # replay through Core ML on this Mac (CPU only: the reference compute unit)
        pred = ml.predict({n: x.numpy() for n, x in zip(in_names, inputs)})
        d = 0.0
        for n, y in zip(out_names, eager):
            y = y.double().numpy()
            s = float(np.abs(y).max()) or 1.0
            d = max(d, float(np.abs(pred[n].astype(np.float64) - y).max()) / s)
        drifts[name] = d
        rec_models[name] = {"file": path.name, "inputs": in_names, "outputs": out_names, "precision": prec}
        print(f"{name}: drift vs eager {d:.2e}", flush=True)

    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    rec = {"format": "coreml-mlprogram", "precision": args.precision, "centred": centred,
           "moment_scale": MOMENT_SCALE if centred else 1.0, "shape": args.shape,
           "seconds_of_audio": args.shape * M.HOP / 16000, "margin": M.MARGIN, "receptive_field": M.RECEPTIVE_FIELD,
           "minimum_deployment_target": args.target, "models": rec_models, "convert_drift": drifts, "bytes": size,
           "export_seconds": round(time.monotonic() - t0, 1), "coremltools": ct.__version__, "torch": torch.__version__}
    (args.out / "export.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
