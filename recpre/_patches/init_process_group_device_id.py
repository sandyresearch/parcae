"""Monkey-patch `torch.distributed.init_process_group` to inject `device_id`
when none is provided and CUDA is available.

PyTorch >=2.4 emits a per-call-site `UserWarning` from `c10d_logger.py:83` of the
form `barrier(): using the device under current context. You can specify
`device_id` in `init_process_group` to mute this warning.` whenever a bare
`torch.distributed.barrier()` is called and the process group was not bound to
a device at init time. Lightning Fabric's `_init_dist_connection` never passes
`device_id`, so even after our user-code barriers all pass `device_ids=...`,
Lightning- and DDP-internal barriers (e.g. during `model.eval()` bucket rebuild,
the very first `validate(...)` call) still trip the warning.

Binding `device_id` at init time makes the process group inherit the device for
every subsequent op, so bare `barrier()` calls anywhere in the stack no longer
warn and PyTorch can also use it to immediately form the NCCL communicator.

Apply once at process start by calling `apply()`.
"""
import os

import torch
import torch.distributed as dist


_APPLIED = False
_ORIGINAL_INIT_PROCESS_GROUP = None


def _resolve_local_rank() -> int:
    for key in ("LOCAL_RANK", "SLURM_LOCALID", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        val = os.environ.get(key)
        if val is not None and val != "":
            try:
                return int(val)
            except ValueError:
                pass
    # Fallback: derive from global rank modulo visible devices. This matches the
    # convention used by SimpleFabric and most single-node launchers.
    rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0"
    try:
        return int(rank) % max(torch.cuda.device_count(), 1)
    except ValueError:
        return 0


def _patched_init_process_group(*args, **kwargs):
    assert _ORIGINAL_INIT_PROCESS_GROUP is not None
    if "device_id" not in kwargs and torch.cuda.is_available():
        kwargs["device_id"] = torch.device("cuda", _resolve_local_rank())
    return _ORIGINAL_INIT_PROCESS_GROUP(*args, **kwargs)


def apply() -> None:
    global _APPLIED, _ORIGINAL_INIT_PROCESS_GROUP
    if _APPLIED:
        return
    _ORIGINAL_INIT_PROCESS_GROUP = dist.init_process_group
    dist.init_process_group = _patched_init_process_group
    _APPLIED = True
