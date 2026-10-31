"""Resume bounded range downloads only from content-verified partial chunks."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .storage import read_json, sha256_file
from .storage import write_json


def download_ranges(
    path,
    size,
    read_range,
    *,
    identity,
    sha256=None,
    block_size=8 * 1024 * 1024,
    workers=16,
):
    """Fill ``path`` using ``read_range(offset, length)`` and atomic receipts.

    Receipts bind source identity and chunk bytes. An old marker cannot certify
    a missing/truncated partial file, and a corrupted chunk is fetched again.
    The caller supplies a pinned full-file SHA when the upstream publishes one.
    """
    path = Path(path)
    if min(size, block_size, workers) <= 0:
        raise ValueError("Download size, block size and workers must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    state_path = path.with_name(path.name + ".resume.json")
    receipts = path.with_name(path.name + ".chunks")
    receipts.mkdir(exist_ok=True)
    state = {"identity": identity, "size": size, "block_size": block_size}
    fingerprint = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    try:
        reusable = (
            path.is_file()
            and path.stat().st_size == size
            and read_json(state_path) == state
        )
    except (OSError, ValueError):
        reusable = False
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, size)
        write_json(state_path, state)

        def fetch(offset):
            length = min(block_size, size - offset)
            marker = receipts / f"{offset}.json"
            if reusable:
                try:
                    saved = read_json(marker)
                    data = os.pread(fd, length, offset)
                    if (
                        saved
                        == {
                            "download_sha256": fingerprint,
                            "offset": offset,
                            "size": length,
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                        and len(data) == length
                    ):
                        return False
                except (OSError, ValueError):
                    pass
            data = read_range(offset, length)
            if len(data) != length:
                raise OSError(f"Truncated download chunk at {offset}")
            written = 0
            while written < length:
                count = os.pwrite(fd, data[written:], offset + written)
                if count <= 0:
                    raise OSError("Could not write downloaded bytes")
                written += count
            os.fsync(fd)
            write_json(
                marker,
                {
                    "download_sha256": fingerprint,
                    "offset": offset,
                    "size": length,
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
            )
            return True

        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched = sum(pool.map(fetch, range(0, size, block_size)))
    finally:
        os.close(fd)
    digest = sha256_file(path)
    if sha256 is not None and digest != sha256:
        # A failed whole-file check invalidates every receipt on the next retry.
        state_path.unlink(missing_ok=True)
        raise OSError("Downloaded file SHA-256 differs from the pinned upstream digest")
    return {"sha256": digest, "fetched_chunks": fetched}
