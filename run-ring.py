#!/usr/bin/env python3
"""Launch four configured ranks over SSH, then supervise them in the foreground."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--inventory',type=Path,required=True)
p.add_argument('--log',type=Path,required=True)
p.add_argument('--mode',choices=('serve','probe'),default='serve')
a=p.parse_args()
site=json.loads(a.inventory.read_text());nodes=site['ranks'];prefix=site['prefix']
if len(nodes)!=4 or not re.fullmatch(r'[A-Za-z0-9_-]+',prefix):p.error('four ranks and a valid prefix required')
for node in nodes:
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+',node['host']) or node['host'].startswith('-'):
        p.error('invalid SSH destination')
    if any(not node[k].startswith('/') for k in ('recipe_dir','env_file','cache_path')):
        p.error('remote paths must be absolute')

def ssh(node,command,timeout):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',node['host'],command],
        text=True,capture_output=True,timeout=timeout)

def check(pair):
    rank,node=pair
    # Host configuration is explicitly trusted operator-owned shell input.
    script='source '+shlex.quote(node['env_file'])+'; '
    script+='[[ "${CONTAINER_PREFIX:-altar}" == '+shlex.quote(prefix)+' && "$CACHE_PATH" == '+shlex.quote(node['cache_path'])+' ]] || exit 2; '
    script+='docker image inspect "$IMAGE_ID" >/dev/null || exit 3; '
    script+='! docker container inspect '+shlex.quote(prefix+'-r'+str(rank))+' >/dev/null 2>&1'
    result=ssh(node,'bash -c '+shlex.quote(script),30)
    if result.returncode:raise RuntimeError(f"{node['host']}: config, image, or existing-container check failed: {result.stderr[-500:]}")

def launch(pair):
    rank,node=pair
    cmd='bash '+shlex.quote(node['recipe_dir']+'/run-rank.sh')+' '+str(rank)+' '+shlex.quote(node['env_file'])+' '+a.mode
    result=ssh(node,cmd,60)
    if result.returncode:raise RuntimeError(f"{node['host']}: launch failed: {result.stderr[-1000:]}")
    print(node['host']+': '+result.stdout.strip(),flush=True)

def stop(pair):
    rank,node=pair
    try:ssh(node,'docker stop -t 5 '+shlex.quote(prefix+'-r'+str(rank)),20)
    except Exception as error:print(f"{node['host']}: stop failed: {error}",file=sys.stderr)

def terminated(signum,frame):raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,terminated)
pairs=list(enumerate(nodes))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(check,pairs))
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,pairs))
    cmd=[sys.executable,str(Path(__file__).with_name('watch_ring.py')),'--prefix',prefix,
        '--hosts',*(n['host'] for n in nodes),'--cache-paths',*(n['cache_path'] for n in nodes),
        '--log',str(a.log)]
    result=subprocess.run(cmd)
    raise SystemExit(result.returncode)
finally:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(stop,pairs))
