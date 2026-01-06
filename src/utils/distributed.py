# path: src/utils/distributed.py
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    is_ddp: bool
    rank: int
    world_size: int
    local_rank: int


def init_distributed() -> DistInfo:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return DistInfo(True, rank, world_size, local_rank)
    return DistInfo(False, 0, 1, 0)


def is_main_process(info: DistInfo) -> bool:
    return info.rank == 0


def barrier_if_ddp(info: DistInfo) -> None:
    if info.is_ddp:
        dist.barrier()


def cleanup_ddp(info: DistInfo) -> None:
    if info.is_ddp:
        dist.destroy_process_group()
