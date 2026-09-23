#!/usr/bin/env python3
"""Fixed-length synthetic-input throughput, separate from model-quality checks."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',required=True)
p.add_argument('--key-file',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--prompt-tokens',type=int,default=8192)
p.add_argument('--output-tokens',type=int,default=256)
p.add_argument('--concurrency',type=int,choices=(1,4),required=True)
p.add_argument('--cache-salt',help='Fresh namespace for a cold run; never reused between runs')
a=p.parse_args()
if a.cache_salt is not None and (not a.cache_salt or len(a.cache_salt)>100 or any(c in a.cache_salt for c in ('\0','\n','\r'))):
    p.error('cache salt must be 1-100 characters without NUL or line breaks')
if a.output.exists():p.error('receipt exists')
headers={'Content-Type':'application/json','Authorization':'Bearer '+a.key_file.read_text().splitlines()[0]}
started=time.monotonic()
def run(rank):
    tokens=([123,456,789,1000]*((a.prompt_tokens+3)//4))[:a.prompt_tokens]
    tokens[0]=1001+rank+a.concurrency
    body={'model':'altar-1','prompt':tokens,'max_tokens':a.output_tokens,'temperature':0,
        'ignore_eos':True,'detokenize':False,'return_token_ids':True,'stream':True,
        'stream_options':{'include_usage':True}}
    if a.cache_salt is not None:body['cache_salt']=a.cache_salt+':'+str(rank)
    req=urllib.request.Request(a.url.rstrip('/')+'/v1/completions',data=json.dumps(body).encode(),headers=headers)
    begin=time.monotonic();first=None;last=None;count=0;done=False;usage=None;finish=None
    with urllib.request.urlopen(req,timeout=7200) as response:
        for line in response:
            text=line.decode().strip()
            if not text.startswith('data: '):continue
            if text[6:]=='[DONE]':done=True;break
            event=json.loads(text[6:]);now=time.monotonic()
            if event.get('usage'):usage=event['usage']
            for choice in event.get('choices',[]):
                ids=choice.get('token_ids') or []
                if ids:
                    if first is None:first=now
                    last=now;count+=len(ids)
                finish=choice.get('finish_reason') or finish
    elapsed=time.monotonic()-begin
    passed=done and count==a.output_tokens and finish=='length' and usage and usage['prompt_tokens']==a.prompt_tokens
    if a.cache_salt is not None:
        passed=passed and (usage.get('prompt_tokens_details') or {}).get('cached_tokens')==0
    return {'rank':rank,'passed':bool(passed),'ttft_s':None if first is None else first-begin,
        'duration_s':elapsed,'generated_tokens':count,'finish_reason':finish,'usage':usage,
        'decode_tokens_per_s':None if first is None or last==first else (count-1)/(last-first)}
rows=[];errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
    futures=[pool.submit(run,rank) for rank in range(a.concurrency)]
    for future in futures:
        try:rows.append(future.result())
        except Exception as error:errors.append(type(error).__name__+': '+str(error))
elapsed=time.monotonic()-started
receipt={'cache_salt_prefix':a.cache_salt,'performance_only':True,'synthetic_input':True,'passed':not errors and all(r['passed'] for r in rows),
    'concurrency':a.concurrency,'prompt_tokens_each':a.prompt_tokens,'output_tokens_each':a.output_tokens,
    'wall_s':elapsed,'aggregate_output_tokens_per_wall_s':sum(r['generated_tokens'] for r in rows)/elapsed,
    'results':rows,'errors':errors}
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt),flush=True)
raise SystemExit(0 if receipt['passed'] else 1)
