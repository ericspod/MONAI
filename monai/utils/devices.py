# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.distributed as dist

__all__ = ["get_default_device", "get_default_rank", "is_rank_0", "get_dist_device"]


def get_default_device(
    device_index: int | None = None, use_rank: bool = False, device: torch.device | str | None = None
) -> torch.device:
    """
    Returns a usable default device for tensors. If CUDA is present, a CUDA device is returned with the selected device
    having index `device_index`, the current rank if `use_rank` is True, or the current device if `use_rank` is False.
    If MPS is available but not CUDA, this device is returned. If neither is available, the CPU device is returned.

    Args:
        device_index: index value of device to use for CUDA or MPS, if None this will be chosen to be the default rank
            if `use_rank` is True (or MPS is selected), otherwise in CUDA the current device is selected.
        use_rank: if True, use the current rank in CUDA to select a device if `device_index` is None, if False (the
            default), use the current device.
        device: if not None, create a device from this value and return it, all other arguments are ignored. This must
            be a device object already or a valid device string for constructing one.
    Returns:
        torch.device object for the selected accelerator device (or CPU if none is found), or the given device.
    """

    if device is not None:
        return torch.device(device)

    device_str = "cpu"
    if torch.cuda.is_available():
        if device_index is not None:
            if device_index < 0 or device_index >= torch.cuda.device_count():
                raise ValueError(f"Invalid argument: {device_index=}, device count is {torch.cuda.device_count()}.")
        elif use_rank:
            device_index = get_default_rank()
        else:
            device_index = torch.cuda.current_device()

        device_str = f"cuda:{int(device_index)}"
    elif torch.backends.mps.is_available():
        if device_index is not None:
            if device_index < 0 or device_index >= torch.mps.device_count():
                raise ValueError(f"Invalid argument: {device_index=}, device count is {torch.mps.device_count()}.")
        else:  # no current_device for MPS so use rank regardless of `use_rank`
            device_index = get_default_rank()

        device_str = f"mps:{int(device_index)}"

    return torch.device(device_str)


def get_default_rank() -> int:
    """
    Returns the current rank if `torch.distributed` is initialized, else 0. This will succeed if distributed is not
    initialized so can be used to choose a default rank or device index in non-distributed runs.
    """
    return dist.get_rank() if dist.is_initialized() else 0


def is_rank_0() -> bool:
    """
    Returns True if the current rank is 0 or if not in a distributed environment, False otherwise. This will succeed in
    non-distributed environments with a return value of True, this can be used as a safe way of determining when things
    should be done for rank 0 or when not using multiprocessing.
    """
    return get_default_rank() == 0


def is_on_cpu(device_or_tensor: torch.device | torch.tensor) -> bool:
    """
    Returns True if `device_or_tensor` is the CPU device, or a tensor allocated in CPU memory, False otherwise.
    """
    if isinstance(device_or_tensor, torch.tensor):
        device_or_tensor = device_or_tensor.device

    return device_or_tensor.type == "cpu"


def get_dist_device(use_rank: bool = False) -> torch.device | None:
    """
    Returns the expected target device in the native PyTorch distributed data parallel environment. For NCCL backend,
    this returns the GPU device of current process (or of current rank if `use_rank` is True). For GLOO backend, this
    returns the CPU device. For other backends, return None as the default, tensor.to(None) will not change the device.

    Args:
        use_rank: if True, use the current rank in CUDA to select a device if `device_index` is None, if False (the
            default), use the current device.

    Returns:
        Torch.device object representing the selected CUDA device if using NCCL, CPU if using gloo, None otherwise.
    """
    backend = dist.get_backend() if dist.is_initialized() else None

    if backend == "nccl" and torch.cuda.is_available():
        return get_default_device(use_rank=use_rank)
    if backend == "gloo":
        return torch.device("cpu")

    return None  # dist not initialized or neither NCCL or gloo backends
