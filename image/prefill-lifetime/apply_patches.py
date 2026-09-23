# SPDX-License-Identifier: MIT
"""Apply pinned preparation-lifetime patches to exactly pinned sources."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, hashes in manifest.items():
        if digest(args.source_dir / name) != hashes["before"]:
            raise SystemExit("Unexpected upstream bytes: " + name)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "1790035200"))
    for name, hashes in manifest.items():
        subprocess.run(
            ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / (name + ".patch"))],
            cwd=args.source_dir, check=True,
        )
        target = args.source_dir / name
        if digest(target) != hashes["after"]:
            raise SystemExit("Patched checksum differs: " + name)
        os.utime(target, (epoch, epoch))
        print("Patched and verified: " + name)
    os.utime(args.source_dir, (epoch, epoch))


if __name__ == "__main__":
    main()
