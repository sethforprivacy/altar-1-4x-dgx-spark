#!/usr/bin/env python3
"""Supervise an already-launched ring; stop its four ranks if one exits or a guard trips."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--prefix',required=True)
p.add_argument('--hosts',nargs=4,required=True)
p.add_argument('--log',type=Path,required=True)
p.add_argument('--cache-paths',nargs=4,required=True,help='Absolute rank-local cache directories')
a=p.parse_args()
if not a.prefix.replace('-','').replace('_','').isalnum():p.error('invalid prefix')
nodes=list(enumerate(a.hosts));misses=[0]*4;seen=[False]*4;started=time.monotonic()

def inspect(pair):
    rank,host=pair;name=f'{a.prefix}-r{rank}'
    guard_path=str(Path(a.cache_paths[rank])/('guard-'+name+'.json'))
    code="import json,subprocess,os,time,re; from pathlib import Path; m=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines()); r=subprocess.run(['docker','inspect','--format','{{.State.Status}}',"+repr(name)+"],capture_output=True,text=True); g=Path('GUARD_PLACEHOLDER'); state=json.loads(g.read_text()); os.kill(state['pid'],0); logs=subprocess.run(['docker','logs','--tail','80','NAME_PLACEHOLDER'],capture_output=True,text=True,timeout=3); fatal=bool(re.search('EngineCore failed to start|WorkerProc failed to start|Worker failed with error|CUDA error:|OutOfMemoryError|NCCL Error',logs.stdout+logs.stderr)); print(json.dumps({'status':r.stdout.strip(),'available_bytes':int(m['MemAvailable'].split()[0])*1024,'guard_age_s':time.time()-g.stat().st_mtime,'fatal_log':fatal,'guard_trip':state.get('trip')}))"
    code=code.replace("'GUARD_PLACEHOLDER'",repr(guard_path)).replace("'NAME_PLACEHOLDER'",repr(name))
    try:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=3',host,
            'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=5,check=True)
        return json.loads(result.stdout)
    except Exception as error:return {'error':type(error).__name__}

def stop(pair):
    rank,host=pair
    cmd='docker stop -t 5 '+shlex.quote(f'{a.prefix}-r{rank}')
    try:subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=3',host,cmd],timeout=15,check=False)
    except subprocess.TimeoutExpired:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=3',host,
            'docker kill '+shlex.quote(f'{a.prefix}-r{rank}')],timeout=10,check=False)

a.log.parent.mkdir(parents=True,exist_ok=True)
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            rows=list(pool.map(inspect,nodes));reason=None
            for rank,row in enumerate(rows):
                running=row.get('status')=='running';seen[rank]|=running
                misses[rank]=misses[rank]+1 if 'error' in row else 0
                if row.get('available_bytes',1<<60)<2*(1<<30):reason='emergency-memory'
                if seen[rank] and not running and 'error' not in row:reason='rank-exited'
                if misses[rank]>=2:reason='rank-unreachable-or-watchdog-dead'
                if row.get('fatal_log'):reason='fatal-engine-error'
                if row.get('guard_trip') or row.get('guard_age_s',0)>10:reason='memory-watchdog-trip-or-stale'
                if time.monotonic()-started>180 and not all(seen):reason='rank-start-timeout'
            with a.log.open('a') as stream:stream.write(json.dumps({'time':time.time(),'ranks':rows,'stop_reason':reason})+'\n')
            if reason:
                list(pool.map(stop,nodes));raise SystemExit(reason)
            time.sleep(2)
except KeyboardInterrupt:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(stop,nodes))
