"""Exercise SparkRing's patched NCCL transport at TP=4; no model is loaded."""
import datetime
import json
import os
import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
assert int(os.environ['WORLD_SIZE']) == 4
torch.cuda.set_device(0)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
for count in (1, 4096, 1048576):
    data = torch.full((count,), rank + 1, device='cuda', dtype=torch.float32)
    dist.all_reduce(data)
    torch.cuda.synchronize()
    assert torch.all(data == 10).item(), (rank, count)
    local=torch.full((count,),rank+1,device='cuda',dtype=torch.float32)
    gathered=torch.empty((4*count,),device='cuda',dtype=torch.float32)
    dist.all_gather_into_tensor(gathered,local)
    for peer in range(4):
        assert torch.all(gathered[peer*count:(peer+1)*count]==peer+1).item()
    reduced=torch.empty_like(local)
    source=torch.full((4*count,),rank+1,device='cuda',dtype=torch.float32)
    dist.reduce_scatter_tensor(reduced,source)
    torch.cuda.synchronize()
    assert torch.all(reduced==10).item()
print(json.dumps({'rank': rank, 'world_size': 4, 'nccl_ring_probe': 'pass',
                  'sizes': [1, 4096, 1048576],
                  'collectives':['all_reduce','all_gather','reduce_scatter']}), flush=True)
dist.destroy_process_group()
