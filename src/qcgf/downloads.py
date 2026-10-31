"""Parallel HTTP-range download that resumes from finished chunks."""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def download_ranges(path, size, read_range, block_size=8 * 1024 * 1024, workers=16):
    """Fill ``path`` with ``read_range(offset, length)``; a marker file per chunk."""
    path = Path(path)
    done = path.with_name(path.name + ".chunks")
    done.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, size)

    def fetch(offset):
        marker = done / str(offset)
        if marker.exists():
            return
        length = min(block_size, size - offset)
        data = read_range(offset, length)
        assert len(data) == length, f"truncated chunk at {offset}"
        os.pwrite(fd, data, offset)
        marker.touch()

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, range(0, size, block_size)))
        os.fsync(fd)
    finally:
        os.close(fd)
    shutil.rmtree(done)
