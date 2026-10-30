"""Gate 4: the on-device host against the Mac host, utterance by utterance.

  python verify/g4_device_vs_mac.py --device _work/device/<UDID>/accuracy.jsonl \
         --mac _runs/g2_w_scan_f512_gpu.json [--out FILE]

Gate 2 established that the Python host on the Mac reproduces NeMo through the exported bundle.  This checks that
the Swift host on the phone does the same thing, on the same NeMo mel, against the same NeMo reference -- so the
two independent implementations of the window schedule and the three rebuilt reductions are compared against each
other as well as against the oracle.

Two different numbers, both reported:
  cos_to_nemo   each host against NeMo (the thing that matters)
  delta         |device cos - mac cos| per utterance (the thing that catches a host bug)
A host bug that happens to be benign on average still shows up in `delta`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=Path, required=True, help="accuracy.jsonl pulled from the phone")
    ap.add_argument("--mac", type=Path, required=True, help="gate 2 json for the same bundle")
    ap.add_argument("--bar", type=float, default=0.9999, help="per-host cosine to NeMo")
    ap.add_argument("--delta-bar", type=float, default=1e-4, help="allowed host-to-host disagreement")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dev = {}
    for line in args.device.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            dev[r["id"]] = r
    mac = {r["id"]: r for r in json.loads(args.mac.read_text())["rows"]}

    both = sorted(set(dev) & set(mac), key=lambda i: mac[i]["seconds"])
    missing = sorted((set(dev) | set(mac)) - set(both))
    if not both:
        sys.exit("no utterances in common between the two runs")

    rows = []
    for i in both:
        d, m = dev[i], mac[i]
        rows.append({"id": i, "seconds": m["seconds"], "n": m["n"], "path": d.get("path", m["path"]),
                     "device_cos": d["cos"], "mac_cos": m["cos"], "delta": abs(d["cos"] - m["cos"]),
                     "device_calls": d.get("calls"), "mac_calls": m.get("calls"),
                     "calls_match": d.get("calls") == m.get("calls")})

    dc = np.array([r["device_cos"] for r in rows], float)
    mc = np.array([r["mac_cos"] for r in rows], float)
    dl = np.array([r["delta"] for r in rows], float)
    worst = rows[int(np.argmax(dl))]
    summary = {"n": len(rows), "missing": missing,
               "device_cos_min": float(dc.min()), "device_cos_mean": float(dc.mean()),
               "mac_cos_min": float(mc.min()), "mac_cos_mean": float(mc.mean()),
               "delta_max": float(dl.max()), "delta_mean": float(dl.mean()),
               "worst_delta_utterance": {k: worst[k] for k in ("id", "seconds", "device_cos", "mac_cos", "delta")},
               "all_calls_match": all(r["calls_match"] for r in rows),
               "device_finite": all(bool(dev[i].get("finite", True)) for i in both)}
    summary["pass"] = bool(summary["device_cos_min"] > args.bar and summary["delta_max"] < args.delta_bar
                           and summary["all_calls_match"] and summary["device_finite"] and not missing)

    print(f"{'id':34s} {'secs':>6s} {'path':>5s} {'device':>12s} {'mac':>12s} {'delta':>10s}")
    for r in rows:
        print(f"{r['id']:34s} {r['seconds']:6.1f} {r['path']:>5s} {r['device_cos']:12.9f} "
              f"{r['mac_cos']:12.9f} {r['delta']:10.2e}")
    print("\n" + json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
