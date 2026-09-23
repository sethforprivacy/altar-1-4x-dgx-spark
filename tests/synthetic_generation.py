#!/usr/bin/env python3
"""Dummy-weight transport/kernel generation only; no semantic quality claim."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://127.0.0.1:8025')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--prompt-tokens',type=int,default=128)
p.add_argument('--output-tokens',type=int,default=32)
p.add_argument('--concurrency',type=int,default=2)
a=p.parse_args()
if a.output.exists():p.error('output receipt already exists')
def test(n):
    body={'model':'altar-1','prompt':([1,123,456,789]*((a.prompt_tokens+3)//4))[:a.prompt_tokens],
        'max_tokens':a.output_tokens,'ignore_eos':True,'temperature':0,'detokenize':False,
        'return_token_ids':True}
    start=time.monotonic()
    request=urllib.request.Request(a.url+'/v1/completions',data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=1800) as response:result=json.load(response)
    assert result['usage']['completion_tokens']==a.output_tokens,result
    return {'request':n,'duration_s':time.monotonic()-start,'usage':result['usage'],
            'finish_reason':result['choices'][0]['finish_reason'],
            'output_token_ids':result['choices'][0].get('token_ids')}
with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
    results=list(pool.map(test,range(a.concurrency)))
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as stream:json.dump({'synthetic_only':True,'passed':True,'results':results},stream,indent=2)
print(json.dumps(results),flush=True)
