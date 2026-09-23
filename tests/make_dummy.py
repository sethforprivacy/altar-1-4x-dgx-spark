#!/usr/bin/env python3
"""Create configuration-only kernel probes; never copy or substitute model weights."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source_config',type=Path)
p.add_argument('output_dir',type=Path)
p.add_argument('--layers',type=int,default=5,help='5 includes dense, BF16 and AWQ experts; 78 tests full depth')
a=p.parse_args()
c=json.loads(a.source_config.read_text())
assert c['architectures']==['GlmMoeDsaForCausalLM'],c['architectures']
assert 5<=a.layers<=c['num_hidden_layers']
a.output_dir.mkdir(parents=True,exist_ok=False)
c['num_hidden_layers']=a.layers
c['num_nextn_predict_layers']=0
for field in ('layer_types','mlp_layer_types','indexer_types'):
    c[field]=c[field][:a.layers]
(a.output_dir/'config.json').write_text(json.dumps(c,indent=2)+'\n')
print('Synthetic configuration only: pass probe mode to the launcher. Outputs have no semantic meaning.')
