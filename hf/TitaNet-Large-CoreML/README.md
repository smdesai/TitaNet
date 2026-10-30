---
license: cc-by-4.0
base_model: nvidia/speakerverification_en_titanet_large
language:
- en
library_name: coreml
tags:
- coreml
- apple
- ios
- macos
- on-device
- neural-engine
- speaker-verification
- speaker-recognition
- speaker-embedding
- titanet
- audio
---

# TitaNet-Large for Core ML

[NVIDIA TitaNet-Large](https://huggingface.co/nvidia/speakerverification_en_titanet_large) (23M parameters) as Core ML
ML Programs for on-device speaker verification and identification on iPhone, iPad and Mac (**iOS 17 / macOS 14 and
later**). Give it 16 kHz mono speech and it returns NeMo's 192-d speaker embedding.

Two variants, each a pair of models (`scan` + `head`) compiled for a single 5.12 s window; a short host loop sweeps
that window over a clip of any length and rebuilds the model's global statistics exactly:

| Variant | Size | Runs on | Cosine to NeMo (worst / mean) | 65 s clip on iPhone | Peak memory on iPhone |
|---|---|---|---|---|---|
| **fp32** | 88.5 MB | GPU, CPU | **0.9999999994** / 0.9999999998 | 0.99 s (GPU) | 64.4 MB |
| **fp16c** | 46.7 MB | Neural Engine, GPU | 0.9968 / 0.9986 (ANE), 0.9997 / 0.9998 (GPU, Mac) | **0.22 s** (ANE) | 95.2 MB |

fp32 is the exact one; fp16c is the fast one, running its `scan` entirely on the Neural Engine. A Core AI build of the
same port (iOS 27 / macOS 27) is at [smdesai/TitaNet-Large-CoreAI](https://huggingface.co/smdesai/TitaNet-Large-CoreAI).

## Results

All comparisons are against NeMo's own embedding of the same audio (the `.nemo` checkpoint, `get_embedding`), over 40
VoxCeleb1-test clips of 4.1–64.7 s.

| Check | Variant · compute | Cosine to NeMo (min / mean) | Notes |
|---|---|---|---|
| WAV → Python host → Core ML (Mac) | fp32 · GPU | **0.9999999994** / 0.9999999998 | the whole pipeline: mel front end, windowed host, models |
| | fp16c · Neural Engine | 0.99684 / 0.99857 | |
| | fp16c · GPU | 0.99966 / 0.99984 | |
| NeMo's mel → Swift host → Core ML (iPhone) | fp32 · GPU | **0.9999999994** / 0.9999999998 | mean 0.36 s per clip, all embeddings finite |
| | fp16c · Neural Engine | 0.99682 / 0.99857 | mean 0.077 s per clip, all embeddings finite |

The iPhone was an iPhone19,4. Memory is the process's `phys_footprint`, the figure iOS's memory limit counts.

What the fp16c differences mean for verification, over all 780 pairs of the 40 clips (score = (cos + 1) / 2):

| Variant · compute | Score error vs NeMo, mean | p99 | worst |
|---|---|---|---|
| fp32 · GPU | 0.00000 | 0.00002 | 0.00004 |
| fp16c · GPU | 0.0006 | 0.0034 | 0.0040 |
| fp16c · Neural Engine | 0.0019 | 0.0083 | 0.0107 |

NVIDIA reports **0.66 % EER on VoxCeleb1 (cleaned trials)** for the base model; this port was not separately
re-scored on that benchmark.

## Files

| Path | What it is |
|---|---|
| `TitaNetScan_fp32.mlpackage`, `TitaNetHead_fp32.mlpackage` | fp32 variant: one window's `scan`, and the embedding `head` |
| `TitaNetScan_fp16c.mlpackage`, `TitaNetHead_fp16c.mlpackage` | fp16c variant: fp16 `scan` (centred attention moments), fp32 `head` |
| `export_fp32.json`, `export_fp16c.json` | window length (512 frames), context margin (46), model signatures, `moment_scale` (fp16c) |
| `frontend/mel_fb.f32` | slaney mel filterbank [80, 257], float32 little-endian (the checkpoint's own buffer) |
| `frontend/mel_window.f32` | Hann window [400], periodic=False (the checkpoint's own buffer) |
| `frontend/config.json` | front-end constants: 16 kHz, n_fft 512, hop 160, win 400, 80 mels, 192-d embedding, threshold 0.7 |

The model files carry the variant in their names so both can sit in one Xcode target: Xcode compiles every
`.mlpackage` to a `.mlmodelc` at the bundle root.

## Input

- **16 kHz, mono**, float samples in [-1, 1]. The model was trained on 16 kHz audio; resample anything else first.
- Any length from one 10 ms frame up. A few seconds of speech gives a stable embedding.
- The front end is NeMo's `AudioToMelSpectrogramPreprocessor` in eval mode, reproduced step for step: pre-emphasis
  0.97, centred 512-point STFT (hop 160, Hann 400), power, 80 slaney mel bins, `log(x + 2^-24)`, per-feature
  normalization over the valid frames. The filterbank and window are the checkpoint's own buffers. Dither is
  training-only and is not applied.

## How it works

### Why one window

One fixed input shape is what the Neural Engine needs, and it bounds memory: the activations of a 5.12 s window are
all the models ever hold, so memory does not grow with the clip (on the iPhone, fp32 sat at 59.5 MB after a 4 s clip
and 59.6 MB after a 65 s one). Longer audio is handled on the host.

### Why a naive window is wrong, and the fix

TitaNet has three reductions over the whole time axis:

1. **five SqueezeExcite gates**: each block scales its channels by `sigmoid(W2 relu(W1 mean_t(x)))` over the entire
   utterance;
2. **the pooling mean / std** of the encoder output;
3. **the attention softmax over time** in attentive statistics pooling.

Computing these per window gives the wrong function: chunk-local SqueezeExcite alone drops the cosine to about 0.36.
The `scan` model makes all three reductions *injectable* (taken as inputs) and *reportable* (partial sums returned as
outputs), so the host can rebuild them exactly:

| Pass | Calls | Rebuilds |
|---|---|---|
| 1–5 | `scan` per window, gate sums fed back as `s0..s4` | the SqueezeExcite fixed point: block *j*'s mean depends only on gates < *j*, so sweep *k* makes gate *k−1* exact and five sweeps make all five exact |
| 6 | `scan` per window | pooling mean / std from running sums `x_sum`, `x_sqsum` |
| 7 | `scan` per window, with the pooling stats | the attention softmax, merged across windows by running maximum (`lmax`, `den` and the attention moments) |
| — | `head` once | the 192-d embedding from the attentive mean and std |

Every convolution is stride 1 with "same" padding and the stack's receptive field is 93 frames, so each window carries
46 frames of real context on each side and keeps only its interior; the interiors tile the clip exactly once. A clip
of *n* mel frames (10 ms each) uses 1 window if *n* ≤ 512, otherwise ⌈(*n* − 512) / 420⌉ + 1, and
**7 × windows + 1** model calls. Host accumulation is float64, and host memory is O(channels), never O(frames).

All five sweeps are needed: the gate residual falls to 6e-4 after two sweeps and *looks* converged while the
embedding cosine is still 0.64. Cosine by number of sweeps: 0.066, 0.639, 0.993, 0.99996, 0.9999987.

### Precision: fp32 and fp16c

**fp32** keeps the checkpoint's arithmetic. Core ML runs fp32 on the GPU or CPU, not the Neural Engine.

**fp16c** is an fp16 `scan` with three exact rewrites, each found by swapping one part of the fp16 model for its fp32
counterpart and measuring the embedding; a plain fp16 conversion reaches only 0.985 (worst) on the Neural Engine:

1. **Centred attention moments.** The attentive variance was formed as `axx/den − (ax/den)²` from fp16 partial sums,
   and that cancellation magnified their rounding. `scan` now reports moments about the pooling mean the host already
   feeds it, `ax_c = Σ w·S(x − mean)` and `axx_c = Σ w·(S(x − mean))²`, so no large terms cancel.
2. **A power-of-two scale** S = 16 (`moment_scale`) on those moments: the encoder output is ~0.02 ± 0.0025, and
   unscaled, 68 % of `axx` came from terms below fp16's smallest normal.
3. **The attention TDNN's BatchNorm gain folded into the convolution before it** (exact because every gain is
   positive, so it passes through the ReLU between them). That convolution's outputs are ~0.08 and the gain reaches
   340; on the Neural Engine that pair's output was off by 5e-2 on average, and by 6e-4 once folded.

The `head` runs in fp32 (one call per clip, on the CPU). What fp16c still loses on the Neural Engine is the encoder's
own fp16 precision: the deviation of its output from the time mean, which the attention variance is made of, is 1.9 %
off on the Neural Engine and 0.2 % on the GPU.

### Interface

`scan`: one window of the clip.

| Input | Shape | Meaning |
|---|---|---|
| `mel` | [1, 80, 512] | the window's normalized log-mel, zero past the clip's end |
| `valid` | [1, 1, 512] | 1 for frames inside the clip |
| `keep` | [1, 1, 512] | 1 for the window's interior frames, whose statistics this call reports |
| `n_inv` | [1, 1] | 1 / (valid frames in the whole clip) |
| `s0`..`s3`, `s4` | [1, 1024] ×4, [1, 3072] | SqueezeExcite sums from the previous sweep (zeros on the first) |
| `mean`, `std` | [1, 3072] each | pooling statistics for the attention pass (zeros and ones before they are known) |

Outputs, all [1, C]: `se0`..`se4` (this window's SqueezeExcite contributions), `x_sum`, `x_sqsum`, `lmax`, `den`, and
the attention moments (C = 3072): `ax`, `axx` for fp32; `ax_c`, `axx_c` for fp16c.

For fp16c, the host sends `mean` already rounded to fp16 in the attention pass, and with S = `moment_scale` and
`q = ax_c / (S · den)` takes the attentive mean as `mean + q` and the variance as `axx_c / (S² · den) − q²`.

`head`: `mu`, `sg` [1, 3072] → `emb` [1, 192].

## License and attribution

This is a derivative of [nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
by NVIDIA, released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); this port is released under the
same license.
Modifications: converted to Core ML, and the network re-authored so its global reductions can be supplied and reported
per window (the attention-pooling concat is folded into its first convolution); the fp16c variant stores its `scan`
weights in fp16, centres and scales its attention moments, and folds the attention BatchNorm's gain into the
convolution before it. The weights are NVIDIA's; no retraining or fine-tuning was done.

```bibtex
@inproceedings{koluguri2022titanet,
  title     = {TitaNet: Neural Model for Speaker Representation with 1D Depth-wise Separable Convolutions and Global Context},
  author    = {Koluguri, Nithin Rao and Park, Taejin and Ginsburg, Boris},
  booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {8102--8106},
  year      = {2022}
}
```
