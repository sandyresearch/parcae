# type: ignore
import torch
from torch.optim import Optimizer
import torch.distributed as dist
import os


class SimpleZeroRedundancyOptimizer(Optimizer):
    """
    Works with the following magic numbers: param group 0 is sharded across the 8 devices on each node.
                                            param group 1 is kept only on device 0
                                            all other groups are not sharded
    """

    verbose = False
    single_rank_return = True

    def __init__(
        self,
        params,
        optimizer_class: Optimizer,
        **optimizer_class_kwargs,
    ):
        # Filter out kwargs not supported by the optimizer
        import inspect
        supported_params = set(inspect.signature(optimizer_class).parameters.keys())
        optimizer_class_kwargs = {k: v for k, v in optimizer_class_kwargs.items() if k in supported_params}
        self.arg_lr = optimizer_class_kwargs["lr"]
        if not torch.distributed.is_initialized():
            self.local_optim_group = optimizer_class(params, **optimizer_class_kwargs)
            self.local_rank = 0
        else:
            global_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
            self.global_rank = global_rank
            self.local_devices = min(int(os.getenv("SLURM_NTASKS_PER_NODE", torch.cuda.device_count())), world_size)
            self.local_rank = int(os.getenv("LOCAL_RANK", global_rank % self.local_devices))
            # Form a local process group
            node_rank = global_rank // self.local_devices
            local_ranks = [rank for rank in range(world_size) if rank // self.local_devices == node_rank]
            self.local_rank_zero_in_global_rank = local_ranks[0]
            if self.verbose:
                print(f"ZERO: Local ranks on {global_rank}: {local_ranks}", flush=True)
            self.local_pg = torch.distributed.new_group(ranks=local_ranks)
            # Assign parameters
            assert self.local_devices > 1
            assert len(params[0]["params"]) > self.local_devices
            assert self.local_rank_zero_in_global_rank + self.local_rank == global_rank
            local_param_groups = []
            if self.local_rank == 0:
                local_param_groups.append(params[1])  # embeddings
                if self.verbose:
                    print(f"ZERO: Placing {len(params[1]['params'])} embedding params on {self.local_rank}")
            else:
                selected_ints = list(range(len(params[0]["params"])))[self.local_rank - 1 :: self.local_devices - 1]
                if self.verbose:
                    print(f"ZERO: Placing weights {selected_ints} on {self.local_rank}")
                local_params = params[0].copy()
                local_params["params"] = params[0]["params"][self.local_rank - 1 :: self.local_devices - 1]
                local_param_groups.append(local_params)
            local_param_groups.extend(params[2:])  # append all higher groups (scalers) to all local ranks

            self.local_optim_group = optimizer_class(local_param_groups, **optimizer_class_kwargs)
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

        super().__init__(params, {})  # this way, super covers all (non-sharded parameters!)
        if self.verbose:
            print(f"ZERO: Optimizer initialized on local_rank {self.local_rank}.")

    @torch.no_grad()
    def step(self, closure=None):
        """Step only on local parameters"""
        self.local_optim_group.step(closure)
        self.sync_parameters()

    @torch.no_grad()
    def sync_parameters(self):
        if torch.distributed.is_initialized():
            # Sync embeddings from rank 0
            for param in self.param_groups[1]["params"]:  # group 1 (embeddings)
                torch.distributed.broadcast(param.data, src=self.local_rank_zero_in_global_rank, group=self.local_pg)

            # Sync weights - each rank broadcasts its chunk
            weight_params = self.param_groups[0]["params"]
            for i in range(self.local_devices - 1):
                for param in weight_params[i :: self.local_devices - 1]:
                    torch.distributed.broadcast(
                        param.data, src=self.local_rank_zero_in_global_rank + i + 1, group=self.local_pg
                    )

    # def zero_grad(self, set_to_none=True):
    """zero_grad automatically executes against all parameters (which is necessary as grads are not sharded)"""

    def state_dict(self, cpu_before_gather=True):
        """Returns the state of the optimizer. Only rank 0 will have the complete state.
        Would it be safer if this didn't even execute on all of the other nodes?"""
        if not torch.distributed.is_initialized():
            return [self._cpu_state_dict(self.local_optim_group.state_dict())]

        if not self.single_rank_return:
            local_state = self.local_optim_group.state_dict()
            return [local_state]
        else:
            torch.cuda.empty_cache()  # thanks HIP
            # Get local state and move to CPU before communication
            local_state = self.local_optim_group.state_dict()
            if cpu_before_gather:
                local_state = self._cpu_state_dict(local_state)
            output_states = [None] * self.local_devices if self.local_rank == 0 else None
            torch.distributed.gather_object(
                local_state, output_states, dst=self.local_rank_zero_in_global_rank, group=self.local_pg
            )
            if not cpu_before_gather:
                output_states = [self._cpu_state_dict(state) for state in output_states]

            return output_states if self.global_rank == 0 else [{"state": {}, "param_groups": []}]

    def load_state_dict(self, state_dict):
        """Each rank loads its relevant parts from the checkpoint."""
        local_devices = getattr(self, 'local_devices', 1)
        local_rank = getattr(self, 'local_rank', 0)
        
        # Detect checkpoint format: single_rank_return=True saves local_devices states,
        # single_rank_return=False saves just 1 state (rank 0 only)
        checkpoint_has_all_ranks = len(state_dict) >= local_devices
        
        if checkpoint_has_all_ranks:
            # Checkpoint was saved with single_rank_return=True - each rank loads its position
            local_state = state_dict[local_rank]
            # Check for empty placeholder state (saved by non-rank-0 processes)
            if not local_state or not local_state.get("state"):
                raise ValueError(
                    f"Checkpoint position {local_rank} has empty state. "
                    f"The checkpoint may have been saved incorrectly."
                )
            print(f"[Rank {local_rank}] Loading optimizer state from position {local_rank}")
            self.local_optim_group.load_state_dict(local_state)
        else:
            # Legacy checkpoint: saved with single_rank_return=False, only rank 0's state exists
            if local_rank == 0:
                print(f"[Rank 0] Loading optimizer state (legacy checkpoint)")
                self.local_optim_group.load_state_dict(state_dict[0])
            else:
                # Ranks 1-7: their optimizer state was never saved in legacy format
                print(f"[Rank {local_rank}] Legacy checkpoint - optimizer state not available, starting fresh")

    def _cpu_state_dict(self, state_dict):
        """Helper to move optimizer state dict to CPU"""
        cpu_state = {}
        for k, v in state_dict.items():
            if k == "state":
                cpu_state[k] = {
                    param_id: {name: val.cpu() if torch.is_tensor(val) else val for name, val in param_state.items()}
                    for param_id, param_state in v.items()
                }
            else:
                cpu_state[k] = v
        return cpu_state

    def __repr__(self):
        return self.__class__.__name__ + self.local_optim_group.__repr__()

    # def __getattr__(self, name):
    #     """Call this only if all other attributes are exhausted."""
    #     return getattr(self.local_optim_group, name)



