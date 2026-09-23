#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the real launcher with Docker/guard spies; no containers or GPUs."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='altar-launcher-') as td:
 work=Path(td);bin_dir=work/'bin';bin_dir.mkdir();record=work/'docker-args.json'
 docker=bin_dir/'docker'
 docker.write_text('#!'+sys.executable+'\n'+'''import json,os,sys
from pathlib import Path
a=sys.argv[1:]
if a[:2]==['image','inspect']:
 if '--format' in a:print(os.environ.get('MOCK_MIXED_LABEL' if 'dsa-ckv-mixed' in a[a.index('--format')+1] else 'MOCK_CKV_LABEL',''))
elif a[:2]==['container','inspect']:raise SystemExit(1)
elif a[0]=='create':Path(os.environ['MOCK_DOCKER_RECORD']).write_text(json.dumps(a));print('c'*64)
elif a[0]=='start':print('started')
else:raise SystemExit('unexpected docker command')
''');docker.chmod(0o755)
 guard=bin_dir/'python3';guard.write_text('#!/bin/sh\nexit 0\n');guard.chmod(0o755)
 base={'IMAGE_ID':'sha256:'+'a'*64,'MODEL_PATH':str(work/'model'),'CACHE_PATH':str(work/'cache'),'HOST_IP':'127.0.0.1','MASTER_ADDR':'127.0.0.1','SOCKET_IFNAME':'lo','NCCL_IB_HCA':'=test:1','DCP_SIZE':'4','MAX_MODEL_LEN':'327680','MAX_NUM_SEQS':'4','MAX_BATCHED_TOKENS':'8192','KV_CACHE_DTYPE':'fp8','CUDA_GRAPH_MODE':'none'}
 checks=[]
 def run(name,overrides,label='1',success=True,enabled=False,mixed_enabled=False,mixed_label='1'):
  record.unlink(missing_ok=True);env_file=work/'rank.env'
  env_file.write_text('\n'.join(k+'='+v for k,v in (base|overrides).items())+'\n')
  env=os.environ|{'PATH':str(bin_dir)+os.pathsep+os.environ['PATH'],'MOCK_DOCKER_RECORD':str(record),'MOCK_CKV_LABEL':label,'MOCK_MIXED_LABEL':mixed_label}
  result=subprocess.run(['bash',str(root/'run-rank.sh'),'0',str(env_file),'probe'],env=env,capture_output=True,text=True)
  assert (result.returncode==0)==success,(name,result.returncode,result.stderr)
  assert record.exists()==success,(name,'unexpected container creation')
  if success:
   args=json.loads(record.read_text())
   assert ('ALTAR_ENABLE_DSA_CKV_MIXED=1' in args)==mixed_enabled
   assert ('ALTAR_ENABLE_DSA_CKV_GATHER=1' in args)==enabled
   assert ('VLLM_B12X_MLA_CKV_GATHER=1' in args)==enabled
   if enabled:
    assert 'VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS=1024' in args
    assert 'VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=1310720' in args
  checks.append(name)
 run('default-off',{})
 run('valid-opt-in',{'EXPERIMENTAL_DSA_CKV_GATHER':'1'},enabled=True)
 run('wrong-image',{'EXPERIMENTAL_DSA_CKV_GATHER':'1'},label='',success=False)
 run('invalid-flag',{'EXPERIMENTAL_DSA_CKV_GATHER':'yes'},success=False)
 for key,value in [('KV_CACHE_DTYPE','nvfp4_ds_mla'),('DCP_SIZE','2'),('MAX_MODEL_LEN','1048576'),('MAX_NUM_SEQS','8'),('MAX_BATCHED_TOKENS','512'),('CUDA_GRAPH_MODE','decode')]:
  run('reject-'+key,{'EXPERIMENTAL_DSA_CKV_GATHER':'1',key:value},success=False)
 run('mixed-valid',{'EXPERIMENTAL_DSA_CKV_GATHER':'1','EXPERIMENTAL_DSA_CKV_MIXED':'1'},enabled=True,mixed_enabled=True)
 run('mixed-requires-base',{'EXPERIMENTAL_DSA_CKV_MIXED':'1'},success=False)
 run('mixed-requires-image',{'EXPERIMENTAL_DSA_CKV_GATHER':'1','EXPERIMENTAL_DSA_CKV_MIXED':'1'},mixed_label='',success=False)
 run('mixed-invalid-flag',{'EXPERIMENTAL_DSA_CKV_MIXED':'yes'},success=False)
 print(json.dumps({'passed':True,'checks':checks,'containers_created':False,'gpu_used':False}))
