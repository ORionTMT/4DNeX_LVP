import gc
from typing import Any, Dict, Optional, Union

import torch
from accelerate.logging import get_logger

from core.finetune.constants import LOG_LEVEL, LOG_NAME


logger = get_logger(LOG_NAME, LOG_LEVEL)


def get_memory_statistics(precision: int = 3) -> Dict[str, Any]:
    memory_allocated = None
    memory_reserved = None
    max_memory_allocated = None
    max_memory_reserved = None

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)

    elif torch.mps.is_available():
        memory_allocated = torch.mps.current_allocated_memory()

    else:
        logger.warning("No CUDA, MPS, or ROCm device found. Memory statistics are not available.")

    memory_allocated_gb = bytes_to_gigabytes(memory_allocated)
    memory_reserved_gb = bytes_to_gigabytes(memory_reserved)
    max_memory_allocated_gb = bytes_to_gigabytes(max_memory_allocated)
    max_memory_reserved_gb = bytes_to_gigabytes(max_memory_reserved)

    return {
        "memory_allocated": round_or_none(memory_allocated_gb, precision),
        "memory_reserved": round_or_none(memory_reserved_gb, precision),
        "max_memory_allocated": round_or_none(max_memory_allocated_gb, precision),
        "max_memory_reserved": round_or_none(max_memory_reserved_gb, precision),
    }


def bytes_to_gigabytes(x: Optional[int]) -> Optional[float]:
    if x is None:
        return None
    return x / 1024**3


def round_or_none(value: Optional[float], precision: int) -> Optional[float]:
    if value is None:
        return None
    return round(value, ndigits=precision)


def free_memory() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # TODO(aryan): handle non-cuda devices


def unload_model(model):
    model.to("cpu")


def make_contiguous(x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(x, torch.Tensor):
        return x.contiguous()
    elif isinstance(x, dict):
        return {k: make_contiguous(v) for k, v in x.items()}
    else:
        return x
