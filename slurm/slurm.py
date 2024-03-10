import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
torch.cuda.set_device(local_rank)
tensor = torch.Tensor([rank]).cuda()
print(f"Before AllReduce: Rank {rank} has value {tensor.item()}")
tensor_ = dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"After AllReduce: Rank {rank} has value {tensor.item()}")
