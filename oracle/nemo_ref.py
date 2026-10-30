"""The publisher's model, driven the way NeMo's own entry points drive it (oracle venv only).

`EncDecSpeakerLabelModel.get_embedding(path)` = infer_file: soundfile read (float64, 16 kHz, no resampling here) ->
float32 tensor at batch 1 -> forward(): preprocessor (AudioToMelSpectrogramPreprocessor, 80 mels, per-feature
normalization, pad_to 16) -> ConvASREncoder (5 JasperBlocks with masked convolutions and SqueezeExcite) ->
SpeakerDecoder (attentive statistics pooling -> BatchNorm -> Conv1d(6144 -> 192) = the embedding; the angular head's
logits). `embed()` runs exactly that chain on one utterance, with NeMo's modules, and returns the tensors the gates
compare; callers assert its embedding equals get_embedding()'s bit for bit.

Scoring is NeMo's: verify_speakers / voxceleb_eval.py take score = (cos(X, Y) + 1) / 2 and decide same-speaker at
score >= 0.7; the EER is voxceleb_eval.py's (sklearn roc_curve, brentq on the interpolated ROC).
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import paths as P  # noqa: E402
NEMO_FILE = P.NEMO_FILE
THRESHOLD = 0.7          # verify_speakers' default


def load_model(dtype=torch.float32):
    import nemo.collections.asr as nemo_asr
    from nemo.utils import logging as nemo_logging
    nemo_logging.setLevel(nemo_logging.ERROR)
    m = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(str(NEMO_FILE), map_location="cpu")
    m.eval()
    return m.to(dtype)


def read_audio(path: Path) -> np.ndarray:
    """float32 samples as infer_file reads a 16 kHz PCM16 WAV (soundfile float64 = int16 / 32768, exact in float32)."""
    x, sr = sf.read(path)
    assert sr == 16000 and x.ndim == 1, (path, sr, x.shape)
    return x.astype(np.float32)


@torch.no_grad()
def embed(model, samples: np.ndarray, layers: bool = False) -> dict:
    """One utterance through forward(). Returns mel, mel_len, the embedding, the logits and (layers=True) the output
    of every encoder block and of the pooling, each cut to the valid frames."""
    model.eval()
    x = torch.from_numpy(samples)[None].to(next(model.parameters()).dtype)
    n = torch.tensor([x.shape[1]])
    mel, mel_len = model.preprocessor(input_signal=x, length=n)
    hooks, blocks, pooled = [], [], []
    if layers:
        for b in model.encoder.encoder:
            hooks.append(b.register_forward_hook(lambda m, i, o: blocks.append(o[0][-1][0].clone())))
        hooks.append(model.decoder._pooling.register_forward_hook(lambda m, i, o: pooled.append(o[0, :, 0].clone())))
    enc, enc_len = model.encoder(audio_signal=mel, length=mel_len)
    logits, emb = model.decoder(encoder_output=enc, length=enc_len)
    for h in hooks:
        h.remove()
    k = int(mel_len)
    out = {"mel": mel[0], "mel_len": k, "emb": emb[0], "logits": logits[0]}
    if layers:
        out["blocks"] = [b[:, :k] for b in blocks]      # 5 x [C, mel_len]
        out["pooled"] = pooled[0]                       # [6144]: cat(mu, sigma)
    return out


def score(a: np.ndarray, b: np.ndarray) -> float:
    """voxceleb_eval.py / verify_speakers: (cos + 1) / 2, in float64 as numpy computes it there."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float((np.dot(a, b) / ((np.dot(a, a) * np.dot(b, b)) ** 0.5) + 1) / 2)


def eer_nemo(scores, labels) -> float:
    """voxceleb_eval.py's EER (percent): sklearn roc_curve, then brentq on 1 - x - interp1d(fpr, tpr)(x)."""
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return 100.0 * brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
