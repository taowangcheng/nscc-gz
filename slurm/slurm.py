import os
import time
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
torch.cuda.set_device(local_rank)
if rank == 0:
    print(f"Init")
tensor = torch.Tensor([rank]).cuda()
print(f"Before AllReduce: Rank {rank} has value {tensor.item()}")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"After AllReduce: Rank {rank} has value {tensor.item()}")

if rank < 8:
    reduce_group = dist.new_group([rank, (rank+8)])
else:
    reduce_group = dist.new_group([rank, (rank-8)])
if rank == 0:
    print(f"Reduce Test")
dist.barrier()
torch.cuda.synchronize()
test_tensor = torch.Tensor([rank]).cuda()
print(f"Before AllReduce: Rank {rank} has value {test_tensor.item()}")
dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM, group=reduce_group)
print(f"After AllReduce: Rank {rank} has value {test_tensor.item()}")

B = 2
S = 4096
H = 11008
if rank == 0:
    print(f"Reduce Benchmark")
reduce_tensor = torch.randn(B, S, H, dtype=torch.float16).cuda()
dist.barrier()
torch.cuda.synchronize()
start = time.time()
dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM, group=reduce_group)
torch.cuda.synchronize()
slot = time.time() - start
speed = B * S * H * 2 / slot / 1024 / 1024 / 1024
print(f"Rank {rank} Time: {slot}s Speed: {speed}GB/s")
