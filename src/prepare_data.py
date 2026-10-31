"""Download a scene-disjoint Grasp-Anything++ subset with HTTP range requests."""

import argparse
import io
import json
import mmap
import os
import pickle
import random
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import requests
import torch

from qcgf.downloads import download_ranges

_http_local = threading.local()
_signed_urls = {}
_disable_signed_cache = set()
_http_lock = threading.Lock()
_cooldown_file = Path("data/.hf_cooldown_until")


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Non-primitive pickle: {module}.{name}")


def primitive_load(data):
    return PrimitiveUnpickler(io.BytesIO(data)).load()


def retry_delay(value):
    """Retry-After permits either seconds or an HTTP-date."""
    try:
        return max(float(value), 1)
    except (TypeError, ValueError):
        try:
            return max(parsedate_to_datetime(value).timestamp() - time.time(), 1)
        except (TypeError, ValueError, OverflowError):
            return 60


def get_range(url, start, end):
    if start < 0 or end < start:
        raise ValueError("A byte range must be nonnegative and nonempty")
    if not hasattr(_http_local, "session"):
        _http_local.session = requests.Session()
    for attempt in range(6):
        try:
            while _cooldown_file.exists():
                try:
                    remaining = float(_cooldown_file.read_text()) - time.time()
                except ValueError:
                    remaining = 1
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 60))
            with _http_lock:
                cached = (
                    _signed_urls.get(url) if url not in _disable_signed_cache else None
                )
            request_url = cached or url + f"?download=true&slice={start}-{end}"
            with _http_local.session.get(
                request_url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=120,
                stream=True,
            ) as r:
                if r.status_code == 429:
                    delay = retry_delay(r.headers.get("Retry-After", 60))
                    _cooldown_file.parent.mkdir(exist_ok=True)
                    with _http_lock:
                        previous = (
                            float(_cooldown_file.read_text())
                            if _cooldown_file.exists()
                            else 0
                        )
                        temporary = Path(
                            str(_cooldown_file)
                            + f".{os.getpid()}.{threading.get_ident()}"
                        )
                        temporary.write_text(
                            str(max(previous, time.time() + delay + 1))
                        )
                        temporary.replace(_cooldown_file)
                    continue
                if cached and r.status_code in {401, 403}:
                    with _http_lock:
                        _signed_urls.pop(url, None)
                    continue
                r.raise_for_status()
                expected = f"bytes {start}-{end}/"
                if r.status_code != 206 or not r.headers.get(
                    "Content-Range", ""
                ).startswith(expected):
                    if cached:
                        with _http_lock:
                            _signed_urls.pop(url, None)
                            _disable_signed_cache.add(url)
                        continue
                    raise OSError(
                        f"Invalid range response: {r.status_code} {r.headers.get('Content-Range')}"
                    )
                if not cached and r.url != request_url:
                    with _http_lock:
                        _signed_urls[url] = r.url
                payload = r.content
                if len(payload) != end - start + 1:
                    raise OSError("Truncated range")
                return payload
        except (OSError, requests.RequestException):
            if attempt == 5:
                raise OSError(
                    f"Range download failed for bytes {start}-{end} after six attempts"
                ) from None
            time.sleep(min(2**attempt, 16))
    raise OSError("HTTP range could not be obtained after retries and rate-limit waits")


class Archive:
    def __init__(self, key, repo, filenames, cache, workers, revision=None):
        self.key, self.workers, self.cache = key, workers, cache
        if revision is None:
            api = requests.get(
                f"https://huggingface.co/api/datasets/{repo}", timeout=30
            )
            api.raise_for_status()
            revision = api.json()["sha"]
        self.revision = revision
        response = requests.get(
            f"https://huggingface.co/api/datasets/{repo}/tree/{self.revision}",
            timeout=30,
        )
        response.raise_for_status()
        tree = response.json()
        sizes = {x["path"]: x["size"] for x in tree}
        self.parts = []
        offset = 0
        for name in filenames:
            self.parts.append(
                (
                    offset,
                    sizes[name],
                    f"https://huggingface.co/datasets/{repo}/resolve/{self.revision}/{name}",
                )
            )
            offset += sizes[name]
        self.size = offset
        self.provenance = {
            "repository": repo,
            "revision": self.revision,
            "files": filenames,
        }

    def read(self, start, size):
        if start < 0 or size < 0 or start + size > self.size:
            raise ValueError("Archive byte range lies outside its parts")
        result = []
        end = start + size
        for offset, length, url in self.parts:
            a, b = max(start, offset), min(end, offset + length)
            if a < b:
                result.append(get_range(url, a - offset, b - offset - 1))
        data = b"".join(result)
        if len(data) != size:
            raise OSError("Invalid archive offset")
        return data

    def index(self, scenes):
        tail_size = min(self.size, 131072)
        tail = self.read(self.size - tail_size, tail_size)
        zi = tail.rfind(b"PK\x06\x06")
        if zi >= 0:
            fields = struct.unpack_from("<4sQ2H2I4Q", tail, zi)
            count, cdsize, offset = fields[7], fields[8], fields[9]
        else:
            end = tail.rfind(b"PK\x05\x06")
            if end < 0:
                raise ValueError("ZIP end-of-directory record is missing")
            fields = struct.unpack_from("<4s4H2IH", tail, end)
            count, cdsize, offset = fields[4:7]
        target = self.cache / f"{self.key}_{self.revision}.cd"
        marker = target.with_suffix(".complete")
        if not marker.exists():
            print(
                f"{self.key}: indexing {count:,} entries ({cdsize / 1e6:.1f} MB)",
                flush=True,
            )
            download_ranges(
                target,
                cdsize,
                lambda pos, length: self.read(offset + pos, length),
                workers=self.workers,
            )
            marker.touch()
        entries = {}
        with (
            target.open("rb") as f,
            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data,
        ):
            pos = 0
            while pos < len(data):
                if data[pos : pos + 4] != b"PK\x01\x02":
                    marker.unlink(missing_ok=True)
                    raise ValueError(f"Invalid central directory at {pos}")
                v = struct.unpack_from("<4s6H3I5H2I", data, pos)
                method, crc, csize, usize = v[4], v[7], v[8], v[9]
                nlen, elen, clen, local = v[10], v[11], v[12], v[16]
                name = data[pos + 46 : pos + 46 + nlen].decode("utf8")
                basename = name.rsplit("/", 1)[-1]
                if basename[:64] in scenes:
                    extra = data[pos + 46 + nlen : pos + 46 + nlen + elen]
                    ep = 0
                    while ep + 4 <= len(extra):
                        tag, length = struct.unpack_from("<HH", extra, ep)
                        if tag == 1:
                            values = iter(
                                struct.unpack_from(
                                    "<" + "Q" * (length // 8), extra, ep + 4
                                )
                            )
                            if usize == 0xFFFFFFFF:
                                usize = next(values)
                            if csize == 0xFFFFFFFF:
                                csize = next(values)
                            if local == 0xFFFFFFFF:
                                local = next(values)
                        ep += 4 + length
                    entries[basename] = {
                        "offset": local,
                        "compressed": csize,
                        "size": usize,
                        "crc": crc,
                        "method": method,
                        "name": name,
                    }
                pos += 46 + nlen + elen + clen
        print(f"{self.key}: found {len(entries):,} relevant entries", flush=True)
        return entries

    def extract(self, entry):
        nlen = len(entry["name"].encode())
        available = min(
            30 + nlen + 256 + entry["compressed"], self.size - entry["offset"]
        )
        block = self.read(entry["offset"], available)
        if block[:4] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        nl, el = struct.unpack_from("<HH", block, 26)
        start = 30 + nl + el
        raw = block[start : start + entry["compressed"]]
        if len(raw) < entry["compressed"]:
            raw = self.read(entry["offset"] + start, entry["compressed"])
        if entry["method"] not in {0, 8}:
            raise ValueError(f"Unsupported ZIP compression method: {entry['method']}")
        payload = zlib.decompress(raw, -15) if entry["method"] == 8 else raw
        if len(payload) != entry["size"] or zlib.crc32(payload) != entry["crc"]:
            raise OSError("ZIP CRC/size mismatch")
        return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"))
    p.add_argument("--train-scenes", type=int, default=1000)
    p.add_argument("--val-scenes", type=int, default=120)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--provenance",
        type=Path,
        help="Pin sources to a prior provenance/dataset_sources JSON",
    )
    args = p.parse_args()
    if min(args.train_scenes, args.val_scenes, args.workers) <= 0:
        p.error("Train/validation scene counts and workers must be positive")
    root = args.root
    cache = root / "metadata"
    cache.mkdir(parents=True, exist_ok=True)
    source_file = args.provenance or Path("configs/dataset_sources.json")
    source_spec = json.loads(source_file.read_text()) if source_file.exists() else {}
    if args.provenance is not None and not source_file.exists():
        raise FileNotFoundError(source_file)
    commit = source_spec.get("split_commit")
    if commit is None:
        commit = requests.get(
            "https://api.github.com/repos/Fsoft-AIC/LGD/commits/main", timeout=30
        ).json()["sha"]
    splits = {}
    for key, path in {
        "train": "train/seen",
        "test_seen": "test/seen",
        "test_unseen": "test/unseen",
    }.items():
        url = f"https://raw.githubusercontent.com/Fsoft-AIC/LGD/{commit}/split/grasp-anything++/{path}.obj"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        splits[key] = set(primitive_load(r.content))
    scene_sets = {key: {x[:64] for x in values} for key, values in splits.items()}
    # A shared scene cannot be used in two evaluation groups or in training.
    shared_test = scene_sets["test_seen"] & scene_sets["test_unseen"]
    scene_sets["test_seen"] -= shared_test
    test_scenes = scene_sets["test_seen"] | scene_sets["test_unseen"]
    eligible = sorted(scene_sets["train"] - test_scenes)
    random.Random(args.seed).shuffle(eligible)
    chosen = {
        "train": set(eligible[: args.train_scenes]),
        "val": set(eligible[args.train_scenes : args.train_scenes + args.val_scenes]),
        "test_seen": scene_sets["test_seen"],
        "test_unseen": scene_sets["test_unseen"],
    }
    scenes = set.union(*chosen.values())
    print("Selected scenes:", {k: len(v) for k, v in chosen.items()}, flush=True)
    archives = {
        "instructions": Archive(
            "instructions",
            "airvlab/Grasp-Anything-pp",
            ["grasp_instructions.zip"],
            cache,
            args.workers,
            source_spec.get("archives", {}).get("instructions", {}).get("revision"),
        ),
        "positive": Archive(
            "positive",
            "airvlab/Grasp-Anything-pp",
            ["grasp_label_positive.zip"],
            cache,
            args.workers,
            source_spec.get("archives", {}).get("positive", {}).get("revision"),
        ),
        "images": Archive(
            "images",
            "airvlab/Grasp-Anything",
            ["image_part_aa", "image_part_ab"],
            cache,
            args.workers,
            source_spec.get("archives", {}).get("images", {}).get("revision"),
        ),
    }
    indices = {k: archive.index(scenes) for k, archive in archives.items()}
    ids = sorted(
        k[:-3]
        for k in indices["positive"]
        if k[:-3] + ".pkl" in indices["instructions"]
        and k[:64] + ".jpg" in indices["images"]
    )
    matched_part_queries = len(ids)
    available_part_scenes = {
        split: len({qid[:64] for qid in ids if qid[:64] in selected})
        for split, selected in chosen.items()
    }
    membership = {}
    for split, selected_scenes in chosen.items():
        allowed = splits["train" if split == "val" else split]
        for qid in ids:
            if qid[:64] in selected_scenes and qid.rsplit("_", 1)[0] in allowed:
                membership[qid] = split
    ids = sorted(membership)
    for directory in ["image", "instructions", "positive"]:
        (root / directory).mkdir(exist_ok=True)
    jobs = [
        ("images", scene + ".jpg", root / "image" / (scene + ".jpg"))
        for scene in sorted({x[:64] for x in ids})
    ]
    jobs += [
        ("instructions", qid + ".pkl", root / "instructions" / (qid + ".pkl"))
        for qid in ids
    ]
    jobs += [
        ("positive", qid + ".pt", root / "positive" / (qid + ".pt")) for qid in ids
    ]

    def download(job):
        key, name, destination = job
        if (
            destination.exists()
            and zlib.crc32(destination.read_bytes()) == indices[key][name]["crc"]
        ):
            return
        payload = archives[key].extract(indices[key][name])
        destination.with_suffix(destination.suffix + ".tmp").write_bytes(payload)
        destination.with_suffix(destination.suffix + ".tmp").replace(destination)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 250 == 0:
                print(f"Downloaded/verified {i}/{len(jobs)}", flush=True)
    manifest = []
    rejected = []
    for qid in ids:
        text = primitive_load((root / "instructions" / (qid + ".pkl")).read_bytes())
        with torch.serialization.safe_globals(
            [
                np.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                type(np.dtype("float64")),
                type(np.dtype("float32")),
            ]
        ):
            labels = torch.load(
                root / "positive" / (qid + ".pt"), map_location="cpu", weights_only=True
            )
        array = np.asarray(labels, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 6:
            rejected.append(
                {"id": qid, "reason": "invalid shape", "shape": list(array.shape)}
            )
            continue
        valid = np.isfinite(array).all(1) & (array[:, 3] > 0) & (array[:, 4] > 0)
        valid &= (array[:, 1:3] >= 0).all(1) & (array[:, 1:3] < 416).all(1)
        array = array[valid]
        if not len(array) or not isinstance(text, str) or not text.strip():
            rejected.append({"id": qid, "reason": "empty annotation"})
            continue
        # Raw convention: [quality, x, y, w, h, theta_degrees], positive image-plane
        # theta rotates clockwise because image y points downward.
        grasps = array[:, 1:].copy()
        grasps[:, 4] = (np.deg2rad(grasps[:, 4]) + np.pi / 2) % np.pi - np.pi / 2
        manifest.append(
            {
                "id": qid,
                "scene_id": qid[:64],
                "object_id": qid.rsplit("_", 1)[0],
                "split": membership[qid],
                "image": "image/" + qid[:64] + ".jpg",
                "text": text.strip(),
                "grasps": grasps.tolist(),
            }
        )
    manifest_bytes = "".join(json.dumps(row) + "\n" for row in manifest).encode()
    temporary = root / "manifest.jsonl.tmp"
    temporary.write_bytes(manifest_bytes)
    temporary.replace(root / "manifest.jsonl")
    metadata = {
        "seed": args.seed,
        "split_commit": commit,
        "archives": {k: a.provenance for k, a in archives.items()},
        "requested_train_scenes": args.train_scenes,
        "requested_val_scenes": args.val_scenes,
        "shared_test_scenes_assigned_to_unseen": len(shared_test),
        "rejected": rejected,
        "candidate_scene_counts": {k: len(v) for k, v in chosen.items()},
        "scenes_with_available_part_labels": available_part_scenes,
        "matched_part_queries_before_object_filter": matched_part_queries,
        "queries_excluded_by_official_object_eligibility": matched_part_queries
        - len(ids),
        "counts": {k: sum(r["split"] == k for r in manifest) for k in chosen},
        "scene_counts": {
            k: len({r["scene_id"] for r in manifest if r["split"] == k}) for k in chosen
        },
    }
    (root / "provenance.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
