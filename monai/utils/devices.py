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
    """
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
        device_str = "mps"

    return torch.device(device_str)


def get_default_rank():
    """
    Returns the current rank if `torch.distributed` is initialized, else 0. This will succeed if distributed is not
    initialized so can be used to choose a default rank or device index in non-distributed runs.
    """
    return dist.get_rank() if dist.is_initialized() else 0


def is_rank_0():
    """
    Returns True if the current rank is 0 or if not in a distributed environment, False otherwise. This will succeed in
    non-distributed environments with a return value of True, this can be used as a safe way of determining when things
    should be done for rank 0 or when not using multiprocessing. 
    """
    return get_default_rank() == 0


def get_dist_device(use_rank: bool = False):
    """
    Get the expected target device in the native PyTorch distributed data parallel.
    For NCCL backend, return GPU device of current process (or of current rank if `use_rank` is True).
    For GLOO backend, return CPU.
    For any other backends, return None as the default, tensor.to(None) will not change the device.
    """
    if dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            return get_default_device(use_rank=use_rank)
        if backend == "gloo":
            return torch.device("cpu")
    return None
