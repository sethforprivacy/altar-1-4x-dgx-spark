#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Execute patched eligibility and routing methods without GPU imports."""
import argparse,ast,hashlib,json,os
from pathlib import Path
from types import SimpleNamespace as NS
r=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--source-path',type=Path,required=True);args=parser.parse_args();p=args.source_path
manifest=json.loads((r/'manifest.json').read_text())['vllm/v1/attention/backends/mla/b12x_mla_sparse.py']
assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['after']
tree=ast.parse(p.read_text());names={'_is_glm_dsa_config','_altar_dsa_ckv_requested','_use_b12x_full_ckv_gather'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_GLM_DSA_MODEL_TYPES' for t in n.targets)]
impl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='B12xMLASparseImpl')
method=next(n for n in impl.body if isinstance(n,ast.FunctionDef) and n.name=='uses_full_ckv_dcp');nodes.append(method)
assign=next(n for n in ast.walk(impl) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='_altar_dsa_mixed_ckv' for t in n.targets))
capture=[False];ns={'os':os,'torch':NS(cuda=NS(is_current_stream_capturing=lambda:capture[0])),'B12xMLASparseMetadata':object}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
checks=[];flags=('ALTAR_ENABLE_DSA_CKV_GATHER','ALTAR_ENABLE_DSA_CKV_MIXED');saved={k:os.environ.get(k) for k in flags}
def check(name,actual,expected):
 assert actual is expected,name
 checks.append(name)
def constructor(model='glm_moe_dsa',dtype='fp8'):
 ns.update(self=NS(),hf_config=NS(model_type=model,kv_lora_rank=512,qk_rope_head_dim=64,index_topk=2048,num_attention_heads=64),kv_cache_dtype=dtype)
 exec(compile(ast.Module(body=[assign],type_ignores=[]),str(p),'exec'),ns)
 return ns['self']._altar_dsa_mixed_ckv
try:
 base=dict(enabled=True,is_glm_next=False,is_glm_dsa=True,dcp_world_size=4,max_query_len=8189,num_tokens=8192,num_decode_tokens=3,min_tokens=1024,max_tokens=1310720)
 eligible=ns['_use_b12x_full_ckv_gather']
 os.environ.pop(flags[1],None);check('mixed-default-off',eligible(**base),False)
 os.environ[flags[1]]='1';os.environ[flags[0]]='0';check('requires-original-opt-in',constructor(),False)
 os.environ[flags[0]]='1';check('both-opt-ins-activate-altar',constructor(),True);check('other-model-not-activated',constructor('glm5_next'),False)
 for n in (1,2,3):check('mixed-decode-rows-'+str(n),eligible(**(base|{'num_decode_tokens':n})),True)
 for key,val in [('enabled',False),('dcp_world_size',1),('max_query_len',1),('num_tokens',1024),('num_tokens',1310721)]:check('fallback-'+key,eligible(**(base|{key:val})),False)
 check('glm-next-mixed-unchanged',eligible(**(base|{'is_glm_dsa':False,'is_glm_next':True})),False)
 check('ordinary-prefill-unchanged',eligible(**(base|{'num_decode_tokens':0})),True)
 check('pure-decode-unchanged',eligible(**(base|{'max_query_len':1,'num_tokens':4,'num_decode_tokens':4})),False)
 for value in ('0','yes',''):
  os.environ[flags[1]]=value;check('only-explicit-one-'+value,eligible(**base),False)
 os.environ[flags[1]]='1'
 fields=dict(num_actual_tokens=8192,num_decode_tokens=3,dcp_ckv_gather_eligible=True,dcp_padded_total_tokens=64,dcp_local_total_tokens=60,ckv_selected_indices=object(),ckv_active_counts=object(),dcp_rank_req_starts=object(),dcp_rank_req_lens=object(),dcp_local_cu_seq_lens=object(),global_cache_seq_lens_per_req=object())
 owner=NS(_ckv_gather_enabled=True,_kernel_page_size_finalized=True,_ckv_local_capacity=128,_altar_dsa_mixed_ckv=True)
 route=ns['uses_full_ckv_dcp'];check('runtime-mixed-route',route(owner,NS(**fields),8192),True)
 for name,val in [('dcp_ckv_gather_eligible',False),('dcp_padded_total_tokens',129),('dcp_padded_total_tokens',0),('dcp_local_total_tokens',65)]:check('runtime-reject-'+name+'-'+str(val),route(owner,NS(**(fields|{name:val})),8192),False)
 for name in ('ckv_selected_indices','ckv_active_counts','dcp_rank_req_starts','dcp_rank_req_lens','dcp_local_cu_seq_lens','global_cache_seq_lens_per_req'):check('runtime-missing-'+name,route(owner,NS(**(fields|{name:None})),8192),False)
 check('runtime-row-mismatch',route(owner,NS(**fields),8191),False)
 for name in ('_ckv_gather_enabled','_kernel_page_size_finalized','_altar_dsa_mixed_ckv'):
  check('runtime-off-'+name,route(NS(**(vars(owner)|{name:False})),NS(**fields),8192),False)
 capture[0]=True;check('capture-stays-fallback',route(owner,NS(**fields),8192),False);capture[0]=False
 del owner._altar_dsa_mixed_ckv;check('legacy-instance-mixed-fallback',route(owner,NS(**fields),8192),False)
 check('legacy-instance-pure-prefill',route(owner,NS(**(fields|{'num_decode_tokens':0})),8192),True)
 print(json.dumps({'passed':True,'adapter_sha256':manifest['after'],'checks':checks,'gpu_imported':False}))
finally:
 for k,v in saved.items():
  if v is None:os.environ.pop(k,None)
  else:os.environ[k]=v
