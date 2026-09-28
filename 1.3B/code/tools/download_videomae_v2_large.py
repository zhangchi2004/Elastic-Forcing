#!/usr/bin/env python3
"""Resume and freeze the official VideoMAE-v2 Large snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import snapshot_download


REPO_ID = "OpenGVLab/VideoMAEv2-Large"
REVISION = "9981a9c8f77118c421e5228e1b219468a4b0238d"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "assets/models/"
            "VideoMAEv2-Large"),
    )
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=str(output),
    )
    files = []
    for path in sorted(output.rglob("*")):
        if (
            not path.is_file()
            or path.name == "download_manifest.json"
            or ".cache" in path.parts
        ):
            continue
        files.append({
            "path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    payload = {
        "schema_version": 1,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "files": files,
    }
    payload["content_hash"] = canonical_sha256(payload)
    atomic_json(output / "download_manifest.json", payload)
    print(json.dumps({
        "output_dir": str(output),
        "file_count": len(files),
        "content_hash": payload["content_hash"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
