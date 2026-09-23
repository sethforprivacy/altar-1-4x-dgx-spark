#!/usr/bin/env python3
"""Check the pinned tokenizer and reasoning headers without loading model weights."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model_dir',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():p.error('receipt exists')
tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
messages=[{'role':'user','content':'Compute 19 + 23.'}]
checks=[]
for effort in ('low','high','max'):
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,reasoning_effort=effort)
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,reasoning_effort=effort)
    # Transformers 5 returns a BatchEncoding; len() would count fields, not tokens.
    if hasattr(ids,'keys'):ids=ids['input_ids']
    checks.append({'effort':effort,'pass':'Reasoning Effort: '+effort.capitalize() in text and len(ids)>4,
                   'tokens':len(ids),'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()})
result={'tokenizer_class':type(tokenizer).__name__,'model_max_length':tokenizer.model_max_length,
        'checks':checks,'passed':all(x['pass'] for x in checks) and tokenizer.model_max_length==1048576}
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['passed'] else 1)
