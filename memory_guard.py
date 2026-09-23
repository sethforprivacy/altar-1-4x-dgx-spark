#!/usr/bin/env python3
"""Original MIT local watchdog: two samples below 6 GiB; immediate kill below 2 GiB."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GIB=1<<30

def decision(available, strikes):
    strikes=strikes+1 if available<6*GIB else 0
    return strikes, 'emergency' if available<2*GIB else ('sustained-pressure' if strikes>=2 else None)

def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable unavailable')

def stop(container, emergency=False):
    try:
        subprocess.run(['docker','kill',container] if emergency else
                       ['docker','stop','-t','5',container],timeout=10,check=False,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','kill',container],timeout=10,check=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--container',required=True)
    ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--daemon',action='store_true')
    args=ap.parse_args()
    args.state.parent.mkdir(parents=True,exist_ok=True)
    if args.daemon:
        with args.state.with_suffix('.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                '--container',args.container,'--state',str(args.state)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        for _ in range(50):
            if child.poll() is not None:raise SystemExit('memory watchdog exited before ready')
            if args.state.exists():
                state=json.loads(args.state.read_text())
                if state['container']==args.container and not state.get('trip'):
                    print(child.pid);return
            time.sleep(.1)
        child.terminate();raise SystemExit('memory watchdog did not become ready')
    strikes=0; failures=0; started=time.monotonic(); seen=False
    while True:
        trip=None
        try:
            mem=available();strikes,trip=decision(mem,strikes);failures=0
        except Exception:
            mem=None;failures+=1
            if failures>=2:trip='memory-probe-unavailable'
        status='emergency-not-inspected'
        if trip!='emergency':
            try:
                result=subprocess.run(['docker','inspect','--format','{{.State.Status}}',args.container],
                    capture_output=True,text=True,timeout=3)
                status=result.stdout.strip() if result.returncode==0 else 'missing'
            except subprocess.TimeoutExpired:status='unknown'
        seen=seen or status=='running'
        state={'utc':datetime.now(timezone.utc).isoformat(),'container':args.container,
            'available_bytes':mem,'strikes':strikes,'trip':trip,'status':status,'pid':os.getpid()}
        temp=args.state.with_suffix('.tmp');temp.write_text(json.dumps(state)+'\n');temp.replace(args.state)
        if trip:
            args.state.with_suffix('.TRIPPED').write_text(json.dumps(state)+'\n')
            stop(args.container,emergency=trip=='emergency');return
        if status in ('exited','dead','missing') and (seen or time.monotonic()-started>60):return
        if not seen and time.monotonic()-started>180:stop(args.container);return
        time.sleep(1)

if __name__=='__main__':main()
