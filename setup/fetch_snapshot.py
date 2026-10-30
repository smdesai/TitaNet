"""Download the TitaNet-Large Hub snapshot into _work/snapshot, unpack the .nemo and record provenance.

Run with either the oracle or the coreai venv: _work/venvs/coreai/bin/python setup/fetch_snapshot.py [--revision SHA]
Writes _work/snapshot/PROVENANCE.json (the resolved revision, every file with its size and sha256) and unpacks the
.nemo tarball (model_config.yaml, model_weights.ckpt) into _work/snapshot/nemo/.
"""
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import paths as P  # noqa: E402
REPO = "nvidia/speakerverification_en_titanet_large"
REVISION = "0dc382f40121a5fbd34db10a2bb04d826c2be6a8"   # main on 2026-10-29 (last change 2023-11-14)
NEMO = "speakerverification_en_titanet_large.nemo"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=REVISION, help="commit sha; 'main' resolves and pins it")
    parser.add_argument("--out", default=str(P.SNAPSHOT))
    args = parser.parse_args()

    revision = HfApi().model_info(REPO).sha if args.revision == "main" else args.revision
    out = Path(args.out)
    snapshot_download(REPO, revision=revision, local_dir=out)

    unpacked = out / "nemo"
    if not (unpacked / "model_weights.ckpt").exists():
        unpacked.mkdir(exist_ok=True)
        with tarfile.open(out / NEMO) as tar:
            for member in tar.getmembers():   # a .nemo is a flat tar: refuse anything that would leave the directory
                assert not member.name.startswith("/") and ".." not in Path(member.name).parts, member.name
            tar.extractall(unpacked, filter="data")

    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and ".cache" not in p.parts and p.name != "PROVENANCE.json":
            files[str(p.relative_to(out))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    record = {"repo": REPO, "revision": revision, "files": files}
    (out / "PROVENANCE.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
