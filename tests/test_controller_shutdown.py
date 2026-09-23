"""Exercise real controller/watcher shutdown with fake SSH and no GPUs or network."""
import json,os,pathlib,signal,subprocess,sys,tempfile,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    folder=pathlib.Path(directory);fake=folder/'ssh';events=folder/'events.jsonl'
    fake.write_text('#!'+sys.executable+'\n'+'''import json,os,sys
command=sys.argv[-1]
kind='inspect' if command.startswith('python3 -c ') else ('stop' if command.startswith('docker stop ') else 'setup')
record={'kind':kind,'parent':os.getppid(),'command':command if kind=='stop' else None}
fd=os.open(os.environ['FAKE_EVENTS'],os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o600)
os.write(fd,(json.dumps(record)+'\\n').encode());os.close(fd)
if kind=='inspect':print(json.dumps({'status':'running','available_bytes':10*(1<<30),'guard_age_s':0,'fatal_log':False,'guard_trip':None}))
''');fake.chmod(0o700)
    inventory=folder/'site.json'
    inventory.write_text(json.dumps({'prefix':'unit-ring','ranks':[{'host':'fake-'+str(i),'recipe_dir':'/fake/recipe','env_file':'/fake/rank.env','cache_path':'/fake/cache'} for i in range(4)]}))
    for index,signum in enumerate((signal.SIGTERM,signal.SIGINT)):
        memory=folder/f'memory-{index}.jsonl'
        proc=subprocess.Popen([sys.executable,str(ROOT/'run-ring.py'),'--inventory',str(inventory),'--log',str(memory)],
            env={**os.environ,'PATH':str(folder)+os.pathsep+os.environ['PATH'],'FAKE_EVENTS':str(events)},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:
            deadline=time.monotonic()+10
            while not memory.exists():
                if proc.poll() is not None:raise AssertionError(proc.communicate())
                if time.monotonic()>deadline:raise AssertionError('watcher never became ready')
                time.sleep(.02)
            rows=[json.loads(x) for x in events.read_text().splitlines()]
            watcher=next(x['parent'] for x in reversed(rows) if x['kind']=='inspect')
            if signum==signal.SIGINT:os.killpg(proc.pid,signum)
            else:proc.send_signal(signum)
            stdout,stderr=proc.communicate(timeout=15)
            try:os.kill(watcher,0)
            except ProcessLookupError:pass
            else:raise AssertionError('old watcher survived controller shutdown')
            rows=[json.loads(x) for x in events.read_text().splitlines()]
            stops=[x for x in rows if x['kind']=='stop' and x['parent']==proc.pid]
            assert len(stops)==4,(stops,stderr)
            assert {x['command'].rsplit('-',1)[1] for x in stops}=={'r0','r1','r2','r3'}
            assert proc.returncode!=0
            print(json.dumps({'signal':int(signum),'watcher_reaped':True,'controller_stopped_all_four_ranks':True}))
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
