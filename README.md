# TitaNet for Core AI and Core ML

[NVIDIA TitaNet-Large](https://huggingface.co/nvidia/speakerverification_en_titanet_large), the 23M-parameter
speaker-verification model, ported to Apple's on-device runtimes: **Core AI** (iOS 27 / macOS 27) and **Core ML**
(iOS 17 / macOS 14). Give it 16 kHz mono speech of any length and it returns NeMo's 192-d speaker embedding.

The converted models are on Hugging Face:

- [smdesai/TitaNet-Large-CoreAI](https://huggingface.co/smdesai/TitaNet-Large-CoreAI): one 47 MB `.aimodel`
- [smdesai/TitaNet-Large-CoreML](https://huggingface.co/smdesai/TitaNet-Large-CoreML): `fp32` (GPU / CPU) and
  `fp16c` (Neural Engine) ML Programs

This repository is everything that produces and verifies them: the re-authored model, the converters, the reference
fixtures from NeMo itself, the verification gates, the standalone hosts and the iOS apps. **[CONVERSION.md](CONVERSION.md)**
walks through it from an empty Mac to verified models.

## Results

Cosine to NeMo's own `get_embedding` over 40 VoxCeleb1-test utterances of 4.1–64.7 s, on an iPhone19,4:

| Model | Runs on | Worst / mean cosine to NeMo | 65 s clip | Peak memory | Size |
|---|---|---|---|---|---|
| Core AI `w_scan_f512` | GPU | 0.9999987 / 0.9999993 | 1.11 s | 117.4 MB | 47 MB |
| Core ML `fp32` | GPU | 0.9999999994 / 0.9999999998 | 0.99 s | 64.4 MB | 89 MB |
| Core ML `fp16c` | Neural Engine | 0.99682 / 0.99857 | 0.22 s | 95.2 MB | 47 MB |

Memory is `phys_footprint`, the figure iOS's memory limit counts. On verification scores, (cos + 1) / 2 over all 780
pairs of the 40 clips, fp16c differs from NeMo by 0.0019 on average and 0.011 at worst. NVIDIA reports 0.66 % EER on
VoxCeleb1 (cleaned) for the base model; the ports were not separately re-scored on it.

## How it works

Every model here is compiled for **one 5.12 s window** (512 mel frames), whatever the length of the audio. A
single fixed shape is what keeps a Core AI bundle small (each function's activation arena is allocated whether or
not it is called, so a ladder of input lengths cost 768 MB) and what the Neural Engine needs.

A naive window is the wrong function, though: TitaNet has three reductions over the whole utterance (five
SqueezeExcite gates, the pooling mean / std, the attention softmax), and leaving even the gates chunk-local drops the
cosine to 0.36. `tnl/model.py` re-authors the network so those reductions are *injectable* and *reportable*, and a
short host loop (`tnl/runtime.py`, `hosts/`) rebuilds them exactly from per-window partial sums: five gate sweeps,
one pooling pass, one attention pass, then the embedding head. Each window carries 46 frames of real context on each
side, so the conv stack's 93-frame receptive field is exact at every kept frame. A 5.12 s window reproduces a 64.7 s
embedding to cos 1.00000000 in fp32.

The Core AI variants that were measured on the phone, for reference (all `scan` + `head`; `wfp16` stores weights in
fp16 and computes in fp32, `fp16` does both in fp16):

| Bundle | Arm | Window | Peak (phone) | Worst cosine to NeMo | Calls @ 64.7 s | Latency |
|---|---|---|---|---|---|---|
| **w_scan_f512** | wfp16 | 512 | **117.4 MB** | **0.9999987** | 113 | 1.11 s |
| w_scan_f256 | wfp16 | 256 | 101.3 MB | 0.9999987 | 274 | 1.61 s |
| v2_scan_f512 | fp16 | 512 | 81.4 MB | 0.9939 | 113 | 0.46 s |
| w_scan_f1024 | wfp16 | 1024 | 129.8 MB | — | 50 | 0.89 s |
| w_both_f512 (+ `once`) | wfp16 | 512 | 174.0 MB | 0.9999987 | 113 | 1.11 s |

`export/export_coreai.py --arm … --shape … --fn …` builds any of them.

## Layout

```
setup/      environments (setup_envs.sh + frozen constraints), checkpoint and evaluation-audio downloads
oracle/     NeMo itself: the reference fixtures every gate compares against
tnl/        the re-authored model (model.py), precision recipes (arms.py), the window schedule (stream.py),
            the Core AI host (runtime.py), and where _work/ lives (paths.py)
export/     export_coreai.py, export_coreml.py, make_assets.py (mel constants), package_hf.py
verify/     g1 graph vs NeMo · g2 Core AI vs NeMo · g3 Mac memory · g4 phone vs Mac · g5 Core ML vs NeMo ·
            g6 Hugging Face repo from WAV vs NeMo · device/ (staging and the on-device gate configs)
hosts/      standalone hosts for the published models: python/ (titanet_coreai.py, titanet_coreml.py) and
            swift/ (coreai/, coreml/)
hf/         the two Hugging Face model cards
apps/       TitaNetStudio (the interactive app), LeanGate (Core AI device gate), MLGate (Core ML device gate)
```

Everything a script produces goes under `_work/` (git-ignored); see `tnl/paths.py`.

## Using the published models

```bash
hf download smdesai/TitaNet-Large-CoreML --local-dir TitaNet-Large-CoreML
python hosts/python/titanet_coreml.py a.wav b.wav --repo TitaNet-Large-CoreML --variant fp16c --compute ane
# score 0.8614 (cos 0.7228): same speaker      (two VoxCeleb1 clips of speaker id10270)
```

`hosts/python/titanet_coreai.py` does the same with the Core AI repository. For an app, the files in
`hosts/swift/coreml` (iOS 17+) or `hosts/swift/coreai` (iOS 27+) are a complete host: mel front end, windowed scan
and a `SpeakerEngine` actor. `apps/TitaNetStudio` shows them in use, downloading each model from Hugging Face on
first use.

## License

The code is MIT ([LICENSE](LICENSE)). The model is NVIDIA's, under CC-BY-4.0; this repository contains none of its
weights (the scripts download them), only the checkpoint's mel filterbank and window for the app's front end.
`apps/TitaNetStudio/Sources/Engine/ModelDownloader.swift` is adapted from Apple's coreai-models sample under
BSD-3-Clause.

```bibtex
@inproceedings{koluguri2022titanet,
  title     = {TitaNet: Neural Model for Speaker Representation with 1D Depth-wise Separable Convolutions and Global Context},
  author    = {Koluguri, Nithin Rao and Park, Taejin and Ginsburg, Boris},
  booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {8102--8106},
  year      = {2022}
}
```
