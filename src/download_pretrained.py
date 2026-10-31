#!/usr/bin/env python3
"""Download pinned CLIP weights and tokenizer, validating every completed file."""

import argparse
import hashlib
from pathlib import Path

import requests
from prepare_data import get_range

from qcgf.storage import sha256_file
from qcgf.downloads import download_ranges
from qcgf.storage import atomic_write, write_json

MODEL = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def git_blob_sha1(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/pretrained/clip"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f"https://huggingface.co/api/models/{MODEL}/tree/{REVISION}", timeout=60
    )
    response.raise_for_status()
    files = {r["path"]: r for r in response.json()}
    name = "pytorch_model.bin"
    entry = files[name]
    size = entry["size"]
    sha = entry["lfs"]["oid"]
    target = root / name
    if not target.exists() or sha256_file(target) != sha:
        partial = target.with_suffix(".partial")
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        download_ranges(
            partial,
            size,
            lambda offset, length: get_range(url, offset, offset + length - 1),
            identity=f"{MODEL}@{REVISION}/{name}:{sha}",
            sha256=sha,
            workers=args.workers,
        )
        partial.replace(target)
    assets = {}
    for name in [
        "config.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
    ]:
        target = root / name
        expected = files[name]["oid"]
        if not target.exists() or git_blob_sha1(target.read_bytes()) != expected:
            r = requests.get(
                f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", timeout=120
            )
            r.raise_for_status()
            if git_blob_sha1(r.content) != expected:
                raise OSError(f"Pinned tokenizer/config checksum differs: {name}")
            atomic_write(target, r.content)
        assets[name] = sha256_file(target)
    write_json(
        root / "provenance.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "pytorch_model_sha256": sha,
            "assets_sha256": assets,
        },
    )
    print("Verified official CLIP weights and tokenizer ready", flush=True)


if __name__ == "__main__":
    main()
