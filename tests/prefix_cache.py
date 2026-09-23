#!/usr/bin/env python3
"""Check prefix reuse and changed-suffix correctness on actual model weights."""
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
a=p.parse_args()
if a.output.exists():p.error('receipt exists')
headers={'Content-Type':'application/json','Authorization':'Bearer '+a.key_file.read_text().splitlines()[0]}
prefix='These inventory rows are reference data. Follow the question after the rows.\n'+''.join(
    f'Item {n}: status normal; reference catalog entry.\n' for n in range(900))

def call(number):
    body={'model':'altar-1','messages':[{'role':'user','content':prefix+
        f'\nCompute 19 + {number}. Reply with only the integer.'}],
        'temperature':0,'reasoning_effort':'low','max_tokens':2048}
    start=time.monotonic()
    req=urllib.request.Request(a.url.rstrip('/')+'/v1/chat/completions',data=json.dumps(body).encode(),headers=headers)
    with urllib.request.urlopen(req,timeout=7200) as response:result=json.load(response)
    choice=result['choices'][0];content=choice['message'].get('content','').strip()
    usage=result['usage'];cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
    return {'number':number,'pass':content==str(19+number) and choice['finish_reason']=='stop',
            'content':content,'usage':usage,'cached_tokens':cached,'duration_s':time.monotonic()-start}

rows=[];errors=[]
try:
    rows.append(call(23));rows.append(call(23));rows.append(call(24))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows.extend(pool.map(call,(25,26)))
except Exception as error:errors.append(type(error).__name__+': '+str(error))
reused=len(rows)==5 and all(r['cached_tokens']>=.75*r['usage']['prompt_tokens'] for r in rows[1:])
passed=len(rows)==5 and all(r['pass'] for r in rows) and reused and not errors
receipt={'passed':passed,'cache_reuse_confirmed':reused,'checks':rows,'errors':errors}
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt),flush=True)
raise SystemExit(0 if passed else 1)
