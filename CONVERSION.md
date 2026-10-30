# Converting TitaNet-Large to Core AI and Core ML

This walks from a clean checkout to verified Core AI and Core ML models, the Hugging Face repositories built from
them, and the on-device checks. Every command runs from the repository root. Everything a step produces goes under
`_work/` (git-ignored; set `TITANET_WORK` to put it elsewhere), so the repository holds only code.

| Step | What | Script | Environment |
|---|---|---|---|
| 1 | Python environments | `setup/setup_envs.sh` | — |
| 2 | NVIDIA's checkpoint | `setup/fetch_snapshot.py` | coreai |
| 3 | Evaluation audio | `setup/fetch_eval_audio.py` | coreai |
| 4 | NeMo reference fixtures | `oracle/make_fixtures.py`, `oracle/make_long_fixtures.py` | oracle |
| 5 | Mel front-end constants | `export/make_assets.py` | coreai |
| 6 | Verify the re-authored graph | `verify/g1_torch_vs_nemo.py` | coreai |
| 7 | Convert to Core AI, verify | `export/export_coreai.py`, `verify/g2_coreai_vs_nemo.py` | coreai |
| 8 | Convert to Core ML, verify | `export/export_coreml.py`, `verify/g5_coreml_vs_nemo.py` | coreml |
| 9 | Package for Hugging Face, verify from WAV | `export/package_hf.py`, `verify/g6_hosts_vs_nemo.py` | coreai, coreml |
| 10 | Verify on an iPhone (optional) | `verify/device/stage.sh`, `apps/LeanGate`, `apps/MLGate` | Xcode |

Every `verify/` script exits non-zero when its check fails.

## Requirements

- An Apple-silicon Mac on **macOS 27** (the Core AI Python runtime runs here; Core ML needs macOS 14 or later).
- [uv](https://docs.astral.sh/uv/) and git. About 15 GB of disk: the NeMo clone, three environments, the evaluation
  audio and the exports.
- For step 10 and the apps: **Xcode 27**, [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`),
  an Apple developer team, and an iPhone on iOS 27 (Core AI) or iOS 18 or later (Core ML).

## 1. Environments

```bash
setup/setup_envs.sh
```

Builds three environments under `_work/venvs`, each installed against a frozen constraints file in `setup/`, so
transitive dependencies are pinned too:

| Environment | Holds | Used for |
|---|---|---|
| `oracle` | NeMo Speech 3.1.0 (cloned at a pinned commit into `_work/ref/nemo`), torch 2.9.0 CPU | the reference: steps 4 only |
| `coreai` | coreai-torch 0.4.3, coreai-core 1.0.0b3, torch 2.9.0 | everything Core AI, plus downloads, assets, fixtures checks |
| `coreml` | coremltools 9.0, torch 2.7.0 (the newest torch coremltools 9.0 is tested with) | everything Core ML |

Re-running the script keeps existing environments and re-applies the pins. The rest of this document writes the
three interpreters as:

```bash
ORACLE=_work/venvs/oracle/bin/python
COREAI=_work/venvs/coreai/bin/python
COREML=_work/venvs/coreml/bin/python
```

## 2. The checkpoint

```bash
$COREAI setup/fetch_snapshot.py
```

Downloads [nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
at a pinned revision (`0dc382f…`) into `_work/snapshot`, unpacks the `.nemo` (`model_config.yaml`,
`model_weights.ckpt`) into `_work/snapshot/nemo/`, and writes `PROVENANCE.json` with every file's sha256. The model
is NVIDIA's, under CC-BY-4.0.

## 3. Evaluation audio

```bash
$COREAI setup/fetch_eval_audio.py
```

Downloads three suites into `_work/eval/<suite>/` as 16 kHz mono PCM16 WAV, each with a `manifest.jsonl`:
VoxCeleb1 test (`vox1_test`, 4,874 utterances, 3.9–69 s) and LibriSpeech test-clean / test-other (`ls_clean`,
`ls_other`), all from pinned Hugging Face dataset revisions, byte for byte. (`--vox1-trials veri_test2.txt` also
installs the official VoxCeleb1-O trial list for an EER run; the steps below do not need it.)

## 4. Reference fixtures

NeMo's own tensors and embeddings, captured once so the `coreai` and `coreml` environments never need NeMo:

```bash
$ORACLE oracle/make_fixtures.py          # _work/fixtures: 120 utterances (40 per suite) with NeMo's mel, its
                                         # pooled [6144] statistics and its embedding                (verify/g1)
$ORACLE oracle/make_long_fixtures.py     # _work/fixtures_long: 40 VoxCeleb1 utterances stratified over 3.9-64.7 s
                                         # with NeMo's mel, mel_len and get_embedding()              (verify/g2, g5, g6)
```

`make_long_fixtures.py` checks that NeMo's forward pass equals `get_embedding()` bit for bit before saving each one.

## 5. Front-end constants

```bash
$COREAI export/make_assets.py
```

Writes `_work/assets/{mel_fb.f32, mel_window.f32, config.json}`: the checkpoint's own mel filterbank [80, 257] and
Hann window [400], bit for bit, and the host constants (16 kHz, n_fft 512, hop 160, 80 mels, 192-d embedding, NeMo's
0.7 threshold). Every host's mel front end reads these; `apps/TitaNetStudio/Resources` carries the same two files.

## 6. Verify the re-authored graph

`tnl/model.py` re-implements TitaNet-Large from the checkpoint with no NeMo import, so that its three whole-utterance
reductions (five SqueezeExcite gates, the pooling statistics, the attention softmax) can be injected and reported
per window. Before converting anything, check it is still the same function:

```bash
$COREAI verify/g1_torch_vs_nemo.py
```

Feeds each fixture's NeMo mel through the graph in float64 and compares the embedding and the pooled statistics with
NeMo's own. **Passes** when both cosines are above 1 − 1e-9 on all 120 utterances and folding the 9216-channel
pooling concat changes nothing (< 1e-9).

## 7. Core AI

```bash
$COREAI export/export_coreai.py --arm wfp16 --shape 512 --fn scan --out _work/bundles/w_scan_f512
$COREAI verify/g2_coreai_vs_nemo.py --bundle _work/bundles/w_scan_f512 --fixtures _work/fixtures_long \
    --compute gpu --out _work/runs/g2_w_scan_f512_gpu.json
```

The export builds one `titanet.aimodel` with two functions for a single 512-frame (5.12 s) window: `scan` (one
window's contribution to all three reductions) and `head` (the embedding from the pooled statistics), plus
`export.json`. It replays each exported program in torch first and refuses to save one that drifts by more than
1e-3. `--arm wfp16` stores weights in fp16 and computes in fp32 (`tnl/arms.py`; full fp16 is not accurate on long
audio). Gate 2 runs the windowed host of `tnl/runtime.py` (five gate sweeps, a pooling pass, an attention pass) on
the Core AI runtime over the 40 long fixtures, fed NeMo's mel. **Passes** when every embedding is finite and the
worst cosine to NeMo is above 0.9999 (measured: 0.9999987).

Optional: `$COREAI verify/g3_mac_peak_memory.py --bundle _work/bundles/w_scan_f512` reports the Mac's peak RSS. It is
only a weak cross-check; the memory figure that matters is the iPhone's, from step 10.

## 8. Core ML

```bash
$COREML export/export_coreml.py --precision fp32  --out _work/coreml/c_scan_f512_fp32
$COREML export/export_coreml.py --precision fp16c --out _work/coreml/c_scan_f512_fp16c
$COREML verify/g5_coreml_vs_nemo.py --bundle _work/coreml/c_scan_f512_fp32  --compute cpu_and_gpu
$COREML verify/g5_coreml_vs_nemo.py --bundle _work/coreml/c_scan_f512_fp16c --compute cpu_and_ne
```

The same `scan` and `head`, traced and converted to ML Programs (iOS 17 / macOS 14) and replayed through Core ML
against torch before saving:

- **fp32** keeps the checkpoint's arithmetic. Core ML runs it on the GPU or CPU.
- **fp16c** is fp16 for the Neural Engine, with three exact rewrites that recover its accuracy (attention moments
  centred on the pooling mean and scaled by 16, the attention BatchNorm's gain folded into the convolution before it)
  and the `head` in fp32. `export/export_coreml.py`'s docstring records what each rewrite fixed and how it was found.

Gate 5 runs the same windowed host through Core ML (`--compute cpu_only | cpu_and_gpu | cpu_and_ne | all`).
**Passes** when every embedding is finite and the worst cosine to NeMo clears the bar for the export's precision:
0.9999 for fp32 (measured 0.99999998 on the GPU), 0.996 for fp16c (measured 0.99682 on the Neural Engine,
0.99966 on the GPU). `--bar` overrides.

## 9. Hugging Face repositories

```bash
$COREAI export/package_hf.py      # _work/hf/TitaNet-Large-CoreAI and _work/hf/TitaNet-Large-CoreML
$COREAI verify/g6_hosts_vs_nemo.py --runtime coreai --repo _work/hf/TitaNet-Large-CoreAI --compute gpu
$COREML verify/g6_hosts_vs_nemo.py --runtime coreml --repo _work/hf/TitaNet-Large-CoreML --variant fp32  --compute gpu
$COREML verify/g6_hosts_vs_nemo.py --runtime coreml --repo _work/hf/TitaNet-Large-CoreML --variant fp16c --compute ane
```

`package_hf.py` lays out the two repositories exactly as published: the model files, their export json, the
`frontend/` constants from step 5, and the model card from `hf/<name>/README.md`. The Core ML files carry their
variant in the name (`TitaNetScan_fp32.mlpackage`, …) so both variants fit one Xcode target.

Gate 6 is the end-to-end check of a repository as a user receives it: each fixture's **WAV** goes through the
standalone host in `hosts/python` (its own numpy mel front end, the windowed host, the repository's models) and the
embedding is compared with NeMo's `get_embedding`. **Passes** when no utterance's frame count differs from NeMo's and
the worst cosine clears 0.9999 (Core AI, Core ML fp32) or 0.996 (fp16c). Point `--repo` at a download
(`hf download smdesai/TitaNet-Large-CoreML --local-dir DIR`) to verify what is actually on the Hub.

Publishing (needs `hf auth login` with write access to the namespace):

```bash
hf upload <namespace>/TitaNet-Large-CoreAI _work/hf/TitaNet-Large-CoreAI . --commit-message "TitaNet-Large for Core AI"
hf upload <namespace>/TitaNet-Large-CoreML _work/hf/TitaNet-Large-CoreML . --commit-message "TitaNet-Large for Core ML"
```

## 10. On an iPhone (optional)

Two headless apps run the same windowed host on the phone, in Swift, against NeMo's embeddings: **LeanGate** (Core
AI) and **MLGate** (Core ML). Each reads a `gate.json`, writes `status.json` (accuracy, latency, and the
`phys_footprint` iOS's memory limit counts) and is driven by its `device.sh` over `devicectl`.

```bash
export DEVELOPMENT_TEAM=<your Apple developer team ID>
export BUNDLE_ID_PREFIX=<e.g. com.yourname>
UDID=<the phone's UDID, from: xcrun devicectl list devices>

verify/device/stage.sh          # NeMo's mel + embeddings for the 40 long fixtures, the Core AI bundle, and the
                                # Core ML models compiled on this Mac, into _work/device_stage/{LeanAssets,MLAssets}

apps/LeanGate/device.sh install $UDID
apps/LeanGate/device.sh push    $UDID
apps/LeanGate/device.sh run     $UDID verify/device/leangate_acc_w_scan_f512.json    # accuracy
apps/LeanGate/device.sh run     $UDID verify/device/leangate_mem_w_scan_f512.json    # memory and latency
$COREAI verify/g4_device_vs_mac.py --device _work/device/$UDID/accuracy.jsonl \
    --mac _work/runs/g2_w_scan_f512_gpu.json       # the phone's Swift host against the Mac's Python host

apps/MLGate/device.sh install $UDID
apps/MLGate/device.sh push    $UDID
apps/MLGate/device.sh run     $UDID verify/device/mlgate_acc_fp32_gpu.json
apps/MLGate/device.sh run     $UDID verify/device/mlgate_acc_fp16c_ane.json
apps/MLGate/device.sh run     $UDID verify/device/mlgate_mem_fp32_gpu.json
apps/MLGate/device.sh run     $UDID verify/device/mlgate_mem_fp16c_ane.json
```

`run` prints the final `status.json`. Accuracy runs report `cos_min` / `cos_mean` over the 40 utterances, which
should match the Mac's gates 2 and 5 (on an iPhone19,4: Core AI 0.9999987, Core ML fp32 0.9999999994, fp16c on the
Neural Engine 0.99682); memory runs report the peak `footprint_mb` (117.4 MB Core AI, 64.4 MB Core ML fp32, 95.2 MB
fp16c) and the latency at 4, 8, 20 and 65 seconds. Results land in `_work/device/$UDID/` (LeanGate) and
`_work/device/$UDID/mlgate/` (MLGate). Gate 4 needs gate 2's `--out` file from step 7.

## The app

`apps/TitaNetStudio` is the interactive app (compare two voices, identify a speaker; microphone or file; Core AI,
Core ML fp32 or fp16c). It ships no model: each runtime's downloads from the Hugging Face repositories on first use.

```bash
export DEVELOPMENT_TEAM=<team ID> BUNDLE_ID_PREFIX=<prefix>
apps/TitaNetStudio/build.sh install $UDID
```

Launched with `-- -selftest YES` it embeds two synthetic clips with every runtime and logs each one's agreement with
Core AI to the console. `apps/TitaNetStudio/make_icon.swift` redraws the app icon.

## Things that will bite you

1. **All five gate sweeps are required.** The SqueezeExcite fixed point is triangular: sweep *k* makes gate *k−1*
   exact. The gate residual falls to 6e-4 after two sweeps and *looks* converged while the embedding cosine is still
   0.64; by sweep count the cosine runs 0.066, 0.639, 0.993, 0.99996, 0.9999987.
2. **`wfp16`, not `fp16`, for Core AI.** A full-fp16 Core AI graph sums thousands of frames' reductions in fp16 and
   collapses on long audio (worst cosine 0.72 at 65 s) while looking fine on 4-second clips.
3. **A plain fp16 Core ML conversion is only 0.985.** Use `--precision fp16c`; see `export/export_coreml.py`.
4. **The Python Core AI host must collect garbage on long runs.** The runtime returns NDArray storage to its pool
   only when Python collects it; `tnl/runtime.py` and `hosts/python/titanet_coreai.py` collect every 32 calls. Swift
   hosts are not affected.
5. **One window, not a ladder of lengths.** In a Core AI bundle every function's activation arena is allocated
   whether or not it is called: a seven-length ladder cost 768 MB to load one function. One 512-frame window keeps
   the whole model at 117 MB on an iPhone, for audio of any length.
