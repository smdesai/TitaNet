---
license: cc-by-4.0
base_model: nvidia/speakerverification_en_titanet_large
language:
- en
library_name: coreai
tags:
- coreai
- apple
- ios
- on-device
- speaker-verification
- speaker-recognition
- speaker-embedding
- titanet
- audio
---

# TitaNet-Large for Core AI

[NVIDIA TitaNet-Large](https://huggingface.co/nvidia/speakerverification_en_titanet_large) (23M parameters) ported
to Apple's **Core AI** runtime for on-device speaker verification and identification on iPhone, iPad and Mac.
Give it 16 kHz mono speech and it returns NeMo's 192-d speaker embedding, numerically matching the original:
**cosine ≥ 0.9999987 against NeMo's `get_embedding` on real recordings from 4 to 65 seconds.**

One small bundle (47 MB) handles audio of any length: the graph is compiled for a single 5.12 s window, and a short
host loop sweeps that window over the clip and rebuilds the model's global statistics exactly. On an iPhone it peaks
at **117 MB** of memory and embeds a 65-second clip in **1.1 s** on the GPU.

## Results

All comparisons are against NeMo's own embedding of the same audio (the `.nemo` checkpoint, `get_embedding`).

| Check | Clips | Cosine to NeMo (min / mean) | Notes |
|---|---|---|---|
| WAV → Python host → Core AI (Mac, GPU) | 40 VoxCeleb1-test, 4.1–64.7 s | **0.9999987** / 0.9999993 | the whole pipeline: mel front end, windowed host, graph |
| NeMo's mel → Swift host → Core AI (iPhone, GPU) | same 40 | **0.9999987** / 0.9999993 | all embeddings finite |

Because the embeddings agree with NeMo's to within 1.3e-6 in cosine, verification behaves as the original does.
NVIDIA reports **0.66 % EER on VoxCeleb1 (cleaned trials)** for the base model; this port was not separately
re-scored on that benchmark.

## Files

| Path | What it is |
|---|---|
| `titanet.aimodel/` | the Core AI bundle: functions `scan` and `head`, fp16 weights, fp32 compute |
| `export.json` | window length (512 frames), context margin (46), function signatures, precision recipe |
| `frontend/mel_fb.f32` | slaney mel filterbank [80, 257], float32 little-endian (the checkpoint's own buffer) |
| `frontend/mel_window.f32` | Hann window [400], periodic=False (the checkpoint's own buffer) |
| `frontend/config.json` | front-end constants: 16 kHz, n_fft 512, hop 160, win 400, 80 mels, 192-d embedding, threshold 0.7 |

## Input

- **16 kHz, mono**, float samples in [-1, 1]. The model was trained on 16 kHz audio; resample anything else first.
- Any length from one 10 ms frame up. A few seconds of speech gives a stable embedding.
- The front end is NeMo's `AudioToMelSpectrogramPreprocessor` in eval mode, reproduced step for step: pre-emphasis
  0.97, centred 512-point STFT (hop 160, Hann 400), power, 80 slaney mel bins, `log(x + 2^-24)`, per-feature
  normalization over the valid frames. The filterbank and window are the checkpoint's own buffers. Dither is
  training-only and is not applied.

## How it works

### Why one window

In a Core AI bundle the activation arena is sized per compiled shape and allocated for every function in the bundle.
A ladder of input-length buckets (the usual way to handle variable-length audio) therefore costs the sum of all its
shapes: 768 MB to over 1.1 GB measured for seven buckets. This port compiles exactly one shape, a 512-frame (5.12 s)
window, and handles longer audio on the host.

### Why a naive window is wrong, and the fix

TitaNet has three reductions over the whole time axis:

1. **five SqueezeExcite gates**: each block scales its channels by `sigmoid(W2 relu(W1 mean_t(x)))` over the entire
   utterance;
2. **the pooling mean / std** of the encoder output;
3. **the attention softmax over time** in attentive statistics pooling.

Computing these per window gives the wrong function: chunk-local SqueezeExcite alone drops the cosine to about 0.36.
The graph here makes all three reductions *injectable* (taken as inputs) and *reportable* (partial sums returned as
outputs), so the host can rebuild them exactly:

| Pass | Calls | Rebuilds |
|---|---|---|
| 1–5 | `scan` per window, gate sums fed back as `s0..s4` | the SqueezeExcite fixed point: block *j*'s mean depends only on gates < *j*, so sweep *k* makes gate *k−1* exact and five sweeps make all five exact |
| 6 | `scan` per window | pooling mean / std from running sums `x_sum`, `x_sqsum` |
| 7 | `scan` per window, with the pooling stats | the attention softmax, merged across windows by running maximum (`lmax`, `den`, `ax`, `axx`) |
| — | `head` once | the 192-d embedding from the attentive mean and std |

Every convolution is stride 1 with "same" padding and the stack's receptive field is 93 frames, so each window carries
46 frames of real context on each side and keeps only its interior; the interiors tile the clip exactly once. A clip
of *n* mel frames (10 ms each) uses 1 window if *n* ≤ 512, otherwise ⌈(*n* − 512) / 420⌉ + 1, and
**7 × windows + 1** model calls. Host accumulation is float64, and host memory is O(channels), never O(frames).

All five sweeps are needed: the gate residual falls to 6e-4 after two sweeps and *looks* converged while the
embedding cosine is still 0.64. Cosine by number of sweeps: 0.066, 0.639, 0.993, 0.99996, 0.9999987.

### Precision

`wfp16`: weights stored as fp16 (half the bytes), all compute in fp32. A full-fp16 graph was rejected for long
audio: its whole-utterance reductions sum thousands of frames in fp16, and the worst-case cosine fell to 0.72 on
clips up to 65 s, although it looked fine (0.99998) on 4-second clips.

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

Outputs, all [1, C]: `se0`..`se4` (this window's SqueezeExcite contributions), `x_sum`, `x_sqsum`, `lmax`, `den`,
`ax`, `axx` (C = 3072).

`head`: `mu`, `sg` [1, 3072] → `emb` [1, 192].

## License and attribution

This is a derivative of [nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
by NVIDIA, released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); this port is released under the
same license.
Modifications: converted to Core AI with fp16 weight storage, and the network re-authored so its global reductions
can be supplied and reported per window (the attention-pooling concat is folded into its first convolution). The
weights are NVIDIA's; no retraining or fine-tuning was done.

```bibtex
@inproceedings{koluguri2022titanet,
  title     = {TitaNet: Neural Model for Speaker Representation with 1D Depth-wise Separable Convolutions and Global Context},
  author    = {Koluguri, Nithin Rao and Park, Taejin and Ginsburg, Boris},
  booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {8102--8106},
  year      = {2022}
}
```
