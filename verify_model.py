#!/usr/bin/env python3
"""Verify the pinned checkpoint against published Hub digests; requires huggingface_hub."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi

REPO = 'AikidoSec/altar-1'
REVISION = '5d591cd98958e4e7e429517887e80ca3bb7ce2c4'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("receipt already exists; choose a new output path")
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise SystemExit('Hub revision mismatch')
    checked = {}
    for entry in info.siblings:
        path = args.model_dir / entry.rfilename
        if not path.is_file() or not path.resolve().is_relative_to(args.model_dir.resolve()):
            raise SystemExit(f'Missing or escaping model file: {entry.rfilename}')
        size = path.stat().st_size
        if entry.size is not None and size != entry.size:
            raise SystemExit(f'Size mismatch: {entry.rfilename}')
        sha = hashlib.sha256()
        blob = hashlib.sha1(f'blob {size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                sha.update(block)
                blob.update(block)
        expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
        actual = sha.hexdigest() if entry.lfs else blob.hexdigest()
        if not expected or actual != expected:
            raise SystemExit(f'Digest mismatch: {entry.rfilename}')
        checked[entry.rfilename] = {'sha256': sha.hexdigest(), 'bytes': size}
        print(f'verified {entry.rfilename}', flush=True)
    result = {'repo': REPO, 'revision': REVISION, 'files': checked,
              'total_bytes': sum(x['bytes'] for x in checked.values())}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')

if __name__ == '__main__':
    main()
