#!/usr/bin/env python3
"""Uncached, one-output-token prefill timing; synthetic performance, not quality."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:8025')
p.add_argument('--key-file', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--depths', default='8192,32768')
p.add_argument('--concurrencies', default='1')
p.add_argument('--repetitions', type=int, default=3)
p.add_argument('--label', required=True)
a = p.parse_args()
if a.output.exists(): p.error('output exists')
depths = [int(x) for x in a.depths.split(',')]
concurrencies = [int(x) for x in a.concurrencies.split(',')]
if any(x < 128 for x in depths) or any(x not in (1,4) for x in concurrencies): p.error('invalid sweep')
headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer '+a.key_file.read_text().splitlines()[0]}
result = {'label': a.label, 'performance_only': True, 'synthetic_input': True,
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'started_at': time.time(), 'warmup_policy': 'One unique uncached request group per shape, retained and excluded from medians.',
          'rows': [], 'phase': 'running'}
a.output.parent.mkdir(parents=True, exist_ok=True)
def save():
    tmp = a.output.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2)); tmp.replace(a.output)

def request(depth, concurrency, iteration, rank):
    # Every shape/run/request starts with a distinct token block. Shared padding
    # after that block cannot produce an automatic prefix-cache hit.
    seed = f'{a.label}:{depth}:{concurrency}:{iteration}:{rank}'
    rng = random.Random(seed)
    tokens = [rng.randrange(1000, 20000) for _ in range(128)]
    tokens += ([123,456,789,1000]*((depth+3)//4))[:depth-128]
    body = {'model':'altar-1', 'prompt':tokens, 'max_tokens':1, 'temperature':0,
            'ignore_eos':True, 'detokenize':False, 'return_token_ids':True,
            'stream':True, 'stream_options':{'include_usage':True}}
    req = urllib.request.Request(a.url.rstrip('/')+'/v1/completions', data=json.dumps(body).encode(), headers=headers)
    start = time.monotonic(); first = None; usage = None; count = 0; done = False; finish = None
    with urllib.request.urlopen(req, timeout=3600) as response:
        for line in response:
            text = line.decode().strip()
            if not text.startswith('data: '): continue
            if text[6:] == '[DONE]': done = True; break
            event = json.loads(text[6:])
            if event.get('usage'): usage = event['usage']
            for choice in event.get('choices', []):
                ids = choice.get('token_ids') or []
                if ids:
                    if first is None: first = time.monotonic()
                    count += len(ids)
                finish = choice.get('finish_reason') or finish
    cached = (usage or {}).get('prompt_tokens_details', {}).get('cached_tokens')
    passed = done and count == 1 and finish == 'length' and cached == 0 and (usage or {}).get('prompt_tokens') == depth
    return {'rank':rank, 'passed':passed, 'request_started_monotonic':start,
            'first_token_monotonic':first, 'ttft_s':None if first is None else first-start,
            'duration_s':time.monotonic()-start, 'usage':usage, 'finish_reason':finish,
            'prompt_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}

save()
try:
    for depth in depths:
        for concurrency in concurrencies:
            for iteration in range(a.repetitions+1):
                started = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(request, depth, concurrency, iteration, rank) for rank in range(concurrency)]
                    rows = [f.result() for f in futures]
                span = max(r['first_token_monotonic'] for r in rows)-min(r['request_started_monotonic'] for r in rows)
                row = {'depth':depth, 'concurrency':concurrency, 'iteration':iteration,
                       'warmup':iteration==0, 'started_at':started, 'ended_at':time.time(),
                       'effective_aggregate_prefill_tokens_per_s':depth*concurrency/span,
                       'last_first_token_s':span, 'passed':all(r['passed'] for r in rows), 'requests':rows}
                result['rows'].append(row); save()
                print(json.dumps({k:v for k,v in row.items() if k!='requests'}), flush=True)
                if not row['passed']: raise RuntimeError('request or zero-cache gate failed')
    result['summary'] = []
    for depth in depths:
        for concurrency in concurrencies:
            rows = [r for r in result['rows'] if r['depth']==depth and r['concurrency']==concurrency and not r['warmup']]
            speeds = [r['effective_aggregate_prefill_tokens_per_s'] for r in rows]
            result['summary'].append({'depth':depth, 'concurrency':concurrency, 'samples':len(rows),
                                      'median_tokens_per_s':statistics.median(speeds), 'min':min(speeds), 'max':max(speeds)})
    result.update(phase='complete', passed=True, ended_at=time.time()); save()
except BaseException as error:
    result.update(phase='failed-or-interrupted', passed=False, error=type(error).__name__+': '+str(error), ended_at=time.time()); save()
    raise
