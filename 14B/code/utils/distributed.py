from datetime import timedelta
from functools import partial
import inspect
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, FullStateDictConfig, FullOptimStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_optim_state_dict(model, optimizer):
    """Gather the full (unsharded) optimizer state on rank 0 (CPU).

    Collective: must be called by every rank.  Returns the full optim state
    dict on rank 0 and an empty/partial dict on other ranks (rank0_only=True).
    """
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=save_policy,
        optim_state_dict_config=optim_policy,
    ):
        osd = FSDP.optim_state_dict(model, optimizer)
    return osd


def fsdp_load_optim_state_dict(model, optimizer, full_osd):
    """Load a full optimizer state dict (present on rank 0) back into ``optimizer``.

    Collective: every rank must call this.  ``full_osd`` need only be non-None on
    rank 0 (FSDP broadcasts/shards it to the other ranks), which keeps host memory
    bounded to a single copy of the optimizer state.
    """
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=save_policy,
        optim_state_dict_config=optim_policy,
    ):
        sharded = FSDP.optim_state_dict_to_load(
            model=model, optim=optimizer, optim_state_dict=full_osd
        )
    optimizer.load_state_dict(sharded)


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False, backward_prefetch="pre"):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        if transformer_module is None:
            raise ValueError("transformer wrap strategy requires transformer_module")
        transformer_classes = (
            {transformer_module}
            if isinstance(transformer_module, type)
            else set(transformer_module)
        )
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_classes
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    backward_prefetch_modes = {
        "pre": BackwardPrefetch.BACKWARD_PRE,
        "post": BackwardPrefetch.BACKWARD_POST,
    }
    if backward_prefetch not in backward_prefetch_modes:
        raise ValueError(
            "FSDP backward_prefetch must be 'pre' or 'post', got "
            f"{backward_prefetch!r}")

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        backward_prefetch=backward_prefetch_modes[backward_prefetch],
        # Every rank constructs and loads the same pretrained/checkpoint state
        # before wrapping; no rank-0 broadcast is implied here.
        sync_module_states=False
    )
    return module


def move_adamw_moment_state_(optimizer, target_device):
    """Move only AdamW moments without changing optimizer values."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("optimizer moment offload requires torch.optim.AdamW")
    target = torch.device(target_device)
    movable_keys = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
    tensor_count = 0
    total_bytes = 0
    initialized_parameters = 0
    for state in optimizer.state.values():
        if not state:
            continue
        initialized_parameters += 1
        unknown = set(state) - movable_keys - {"step"}
        if unknown:
            raise RuntimeError(
                "AdamW state contains unsupported entries for exact offload: "
                f"{sorted(unknown)}")
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            raise RuntimeError(
                "initialized AdamW state is missing moment buffers")
        step = state.get("step")
        if not torch.is_tensor(step) or step.device.type != "cpu":
            raise RuntimeError(
                "non-capturable AdamW step state must remain on CPU")
        for key in movable_keys:
            value = state.get(key)
            if value is None:
                continue
            if not torch.is_tensor(value):
                raise RuntimeError(f"AdamW {key} state is not a tensor")
            if value.device != target:
                value = value.to(device=target, non_blocking=False)
                state[key] = value
            if value.device != target:
                raise RuntimeError(
                    f"AdamW {key} state did not move to {target}")
            tensor_count += 1
            total_bytes += value.numel() * value.element_size()
    return {
        "target_device": str(target),
        "initialized_parameters": initialized_parameters,
        "tensor_count": tensor_count,
        "bytes": total_bytes,
    }


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])
    # Bind CUDA before NCCL creates communicators.
    torch.cuda.set_device(local_rank)

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    init_kwargs = {}
    if (
        backend == "nccl"
        and "device_id" in inspect.signature(
            dist.init_process_group).parameters
    ):
        init_kwargs["device_id"] = torch.device("cuda", local_rank)
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=30),
        **init_kwargs,
    )


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
