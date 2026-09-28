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


def get_default_device(device_index: int | None = None, use_rank: bool = False):
    """
    Returns a usable default device for tensors. If CUDA is present, a CUDA device is returned with the selected device
    having index `device_index`, the current rank if `use_rank` is True, or the current device if `use_rank` is False.
    If MPS is available but not CUDA, this device is returned. If neither is available, the CPU device is returned.

    Args:
        device_index: selected CUDA device index, If None, rank is chosen if `use_rank` otherwise current device.
        use_rank: If True, default device index will be rank, or if False (the default) the current device.

    Returns:
        A PyTorch device object, CUDA if available, MPS if no CUDA, CPU if neither.

    Raises:
        ValueError: if `device_index` is given and not an int, or outside the range [0, torch.cuda.device_count() - 1].
    """
    device_str = "cpu"

    # always check for correct argumentvalue even if CUDA isn't available
    if device_index is not None:
        if not isinstance(device_index, int) or device_index < 0 or device_index >= torch.cuda.device_count():
            raise ValueError(f"Invalid argument: {device_index=}, device count is {torch.cuda.device_count()}.")

    if torch.cuda.is_available():
        if device_index is None:
            device_index = get_default_rank() if use_rank else torch.cuda.current_device()

        device_str = f"cuda:{int(device_index)}"
    elif torch.backends.mps.is_available():
        device_str = "mps"

    return torch.device(device_str)


def get_default_rank():
    """
    Returns the current rank if `torch.distributed` is initialized, else 0. This will succeed if distributed is not
    initialized so can be used to choose a default rank or device index in non-distributed runs.

    Returns:
        0 if torch.distributed is not initialized, otherwise the current rank (which may be 0).
    """
    return dist.get_rank() if dist.is_initialized() else 0


def is_rank_0():
    """
    Returns True if the current rank is 0 or if not in a distributed environment, False otherwise. This will succeed in
    non-distributed environments with a return value of True, this can be used as a safe way of determining when to do
    things on rank 0 or when not using multiprocessing. For example, use this to disable logging on other ranks.

    Returns:
        True if the current rank is 0 or torch.distributed isn't initialized, False otherwise.
    """
    return get_default_rank() == 0


def get_dist_device(use_rank: bool = False):
    """
    Returns the expected target device in the native PyTorch distributed data parallel based on backend. For the NCCL
    backend, returns the current device of current process (or of the current rank if `use_rank` is True). For the GLOO
    backend, or if CUDA isn't available, returns the CPU device. For any other backends or if torch.distributed isn't
    initialized, returns None as the default which is safe as `tensor.to(None)` will not change a tensor's device.

    Arg:
        use_rank: If True, select the current rank for the device index if a CUDA device is returned.

    Returns:
        The default device if NCCL is the backend and CUDA is available, CPU device if GLOO, None otherwise.
    """
    if dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            return get_default_device(use_rank=use_rank)
        if backend == "gloo":
            return torch.device("cpu")
    return None
