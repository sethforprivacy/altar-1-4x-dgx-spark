"""Apply pinned compatibility patches; reject unknown source bytes."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'patches/manifest.json').read_text())
site = Path('/opt/venv/lib/python3.12/site-packages')
epoch = int(os.environ.get('SOURCE_DATE_EPOCH', '1790035200'))
for entry in manifest:
    target = site / entry['file']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(target) != entry['before_sha256']:
        raise SystemExit('Unexpected upstream bytes: ' + entry['file'])
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i',
                    str(root / 'patches' / entry['patch'])], cwd=site, check=True)
    if digest(target) != entry['after_sha256']:
        raise SystemExit('Patched checksum differs: ' + entry['file'])
    os.utime(target, (epoch, epoch))
    os.utime(target.parent, (epoch, epoch))
    print('Patched and verified: ' + entry['file'])
