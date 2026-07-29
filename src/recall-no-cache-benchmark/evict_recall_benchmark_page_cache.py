#!/usr/bin/env python3
"""Evict only scratch benchmark files from the Linux page cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def evict_file_page_cache(path: Path) -> int:
    """Advise Linux that one closed scratch file's cached pages are not needed."""
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        return metadata.st_size
    finally:
        os.close(descriptor)


def iter_regular_files(root: Path):
    """Yield regular non-symlink files beneath one scratch path."""
    if root.is_file() and not root.is_symlink():
        yield root
        return
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not (Path(directory) / name).is_symlink()
        )
        for file_name in sorted(file_names):
            path = Path(directory) / file_name
            if path.is_file() and not path.is_symlink():
                yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evict private recall benchmark files with POSIX_FADV_DONTNEED."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args()

    file_count = 0
    byte_count = 0
    for root in arguments.paths:
        resolved = root.expanduser().resolve(strict=True)
        for path in iter_regular_files(resolved):
            byte_count += evict_file_page_cache(path)
            file_count += 1

    print(json.dumps({"files": file_count, "bytes": byte_count}, separators=(",", ":")))


if __name__ == "__main__":
    main()
