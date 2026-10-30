"""Download the evaluation audio into _work/eval/<suite>/ as 16 kHz mono PCM16 WAV, plus manifest.jsonl and trials.

  _work/venvs/coreai/bin/python setup/fetch_eval_audio.py [--suites vox1_test,ls_clean,ls_other]

Suites:
  vox1_test            VoxCeleb1 test, the 40 speakers / 4,874 utterances behind the card's VoxCeleb1-O EER
                       (Codec-SUPERB/Voxceleb1_test_original, pinned: the original WAVs, kept byte for byte)
  ls_clean, ls_other   LibriSpeech test-clean / test-other (openslr/librispeech_asr, pinned), 40 / 33 speakers
Each manifest line: {"id", "audio" (path relative to the suite directory), "seconds", "speaker"}.
trials.txt is "<1 = same speaker | 0> <id> <id>" per line, as VoxCeleb's veri_test2.txt. For vox1_test it is the
official cleaned VoxCeleb1-O list (the card's metric), copied in by hand (--vox1-trials PATH, sha256 checked): it is
served only from robots.ox.ac.uk. For LibriSpeech the trials are drawn here with a fixed seed: every utterance is
paired with one other utterance of its speaker and one of another speaker.
Nothing is resampled or re-encoded, so every host and the oracle read identical samples. The downloads are verified
against the Hub's sha256 and kept under _work/eval/_hub.
"""
import argparse
import hashlib
import io
import json
import random
import shutil
import sys
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import paths as P  # noqa: E402
OUT = P.EVAL
LS_REPO, LS_REV = "openslr/librispeech_asr", "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
VOX_REPO, VOX_REV = "Codec-SUPERB/Voxceleb1_test_original", "1123bd76eb6064ff648bb3c9d4eb75c42e432e7a"
VOX_SHARDS = ["data/test-00000-of-00003-d0eea6966e0eba2f.parquet", "data/test-00001-of-00003-1ab27c4d03da1754.parquet",
              "data/test-00002-of-00003-4d6815b5778a4368.parquet"]
# https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt (VoxCeleb1-O cleaned, 37,611 trials)
VERI_TEST2_SHA256 = None   # pinned on first copy: see trials_vox1()
DEFAULT = "vox1_test,ls_clean,ls_other"


def write_manifest(d: Path, rows: list) -> None:
    rows.sort(key=lambda r: r["id"])
    (d / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    n_spk = len({r["speaker"] for r in rows})
    print(f"{d.name}: {len(rows)} utterances, {n_spk} speakers, {sum(r['seconds'] for r in rows) / 3600:.2f} h",
          flush=True)


def vox1_test(suite: str) -> None:
    import pyarrow.parquet as pq
    d = OUT / suite
    rows = []
    for shard in VOX_SHARDS:
        p = hf_hub_download(VOX_REPO, shard, repo_type="dataset", revision=VOX_REV, cache_dir=OUT / "_hub")
        for batch in pq.ParquetFile(p).iter_batches(batch_size=64):
            for r in batch.to_pylist():
                # id10270+5r0dWxy17C8+00001.wav -> id10270/5r0dWxy17C8/00001, the trial list's own naming
                uid = r["id"].removesuffix(".wav").replace("+", "/")
                raw = r["audio"]["bytes"]
                info = sf.info(io.BytesIO(raw))
                assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16", (uid, info)
                wav = d / "wav" / f"{uid}.wav"
                if not wav.exists():
                    wav.parent.mkdir(parents=True, exist_ok=True)
                    wav.write_bytes(raw)
                rows.append({"id": uid, "audio": f"wav/{uid}.wav", "seconds": info.frames / 16000,
                             "speaker": uid.split("/")[0]})
    write_manifest(d, rows)


def librispeech(suite: str) -> None:
    import pyarrow.parquet as pq
    split = {"ls_clean": "clean/test/0000.parquet", "ls_other": "other/test/0000.parquet"}[suite]
    p = hf_hub_download(LS_REPO, split, repo_type="dataset", revision=LS_REV, cache_dir=OUT / "_hub")
    d = OUT / suite
    (d / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in pq.ParquetFile(p).iter_batches(batch_size=64):
        for r in batch.to_pylist():
            # the FLAC holds the original PCM16 samples; read them as int16 so the WAV is bit-identical
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")
            assert sr == 16000 and x.ndim == 1, (r["id"], sr, x.shape)
            wav = d / "wav" / f"{r['id']}.wav"
            if not wav.exists():
                sf.write(wav, x, sr, subtype="PCM_16")
            rows.append({"id": r["id"], "audio": f"wav/{r['id']}.wav", "seconds": len(x) / sr,
                         "speaker": str(r["speaker_id"])})
    write_manifest(d, rows)
    trials_drawn(d, rows)


def trials_drawn(d: Path, rows: list, seed: int = 0) -> None:
    """One target and one non-target trial per utterance, drawn with a fixed seed; no utterance paired with itself."""
    rng = random.Random(seed)
    by_spk = {}
    for r in rows:
        by_spk.setdefault(r["speaker"], []).append(r["id"])
    speakers = sorted(by_spk)
    lines = []
    for r in rows:
        same = [u for u in by_spk[r["speaker"]] if u != r["id"]]
        lines.append(f"1 {r['id']} {rng.choice(same)}")
        other = rng.choice([s for s in speakers if s != r["speaker"]])
        lines.append(f"0 {r['id']} {rng.choice(by_spk[other])}")
    (d / "trials.txt").write_text("\n".join(lines) + "\n")
    print(f"{d.name}: {len(lines)} trials drawn (seed {seed})", flush=True)


def trials_vox1(src: Path) -> None:
    d = OUT / "vox1_test"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    if VERI_TEST2_SHA256 is not None:
        assert digest == VERI_TEST2_SHA256, f"{src}: sha256 {digest}, expected {VERI_TEST2_SHA256}"
    ids = {json.loads(line)["id"] for line in (d / "manifest.jsonl").open()}
    lines = [line.split() for line in src.read_text().splitlines() if line.strip()]
    for label, a, b in lines:
        assert label in "01" and a.removesuffix(".wav") in ids and b.removesuffix(".wav") in ids, (label, a, b)
    (d / "trials.txt").write_text("".join(f"{lab} {a.removesuffix('.wav')} {b.removesuffix('.wav')}\n"
                                          for lab, a, b in lines))
    shutil.copyfile(src, d / "veri_test2.txt")
    print(f"vox1_test: {len(lines)} trials ({sum(l[0] == '1' for l in lines)} target), sha256 {digest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default=DEFAULT)
    ap.add_argument("--vox1-trials", default=None, help="path to veri_test2.txt (VoxCeleb1-O cleaned)")
    args = ap.parse_args()
    for suite in args.suites.split(","):
        if (OUT / suite / "manifest.jsonl").exists():
            print(f"{suite}: present", flush=True)
            continue
        vox1_test(suite) if suite == "vox1_test" else librispeech(suite)
    if args.vox1_trials:
        trials_vox1(Path(args.vox1_trials))


if __name__ == "__main__":
    main()
