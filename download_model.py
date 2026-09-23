#!/usr/bin/env python3
"""Download the pinned checkpoint, then verify every published file digest."""
import argparse
from pathlib import Path
import subprocess
import sys
from huggingface_hub import snapshot_download
from verify_model import REPO, REVISION

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model_dir',type=Path)
p.add_argument('--receipt',type=Path,required=True)
a=p.parse_args()
if a.receipt.exists():p.error('receipt exists; select a new path')
snapshot_download(REPO,revision=REVISION,local_dir=a.model_dir,max_workers=8)
subprocess.run([sys.executable,str(Path(__file__).with_name('verify_model.py')),
                str(a.model_dir),'--receipt',str(a.receipt)],check=True)
