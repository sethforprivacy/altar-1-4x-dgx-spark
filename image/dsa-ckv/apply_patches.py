# SPDX-License-Identifier: MIT
"""Apply the opt-in full-GLM CKV patch only to exact lifetime-patched sources."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-dir',type=Path,default=Path('/opt/venv/lib/python3.12/site-packages'))
a=p.parse_args();root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,hashes in manifest.items():
 if digest(a.source_dir/name)!=hashes['before']:raise SystemExit('Unexpected upstream bytes: '+name)
epoch=int(os.environ.get('SOURCE_DATE_EPOCH','1790035200'))
for name,hashes in manifest.items():
 patch=root/(Path(name).name+'.patch')
 subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(patch)],cwd=a.source_dir,check=True)
 target=a.source_dir/name
 if digest(target)!=hashes['after']:raise SystemExit('Patched checksum differs: '+name)
 os.utime(target,(epoch,epoch));os.utime(target.parent,(epoch,epoch))
 print('Patched and verified: '+name)
