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

from itertools import chain
import os
import re
import torch
import torch.distributed as dist

__all__ = [
    "get_default_device",
    "get_default_rank",
    "find_device",
    "is_rank_0",
    "is_on_cpu",
    "get_dist_device",
    "count_env_devices",
    "CUDA_VISIBLE_DEVICES",
]

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


def get_default_device(
    device_idx: int | None = None, use_rank: bool = False, device: torch.device | str | None = None
) -> torch.device:
    """
    Returns a usable default device for tensors. If CUDA is present, a CUDA device is returned with the selected device
    having index `device_idx`, the current rank if `use_rank` is True, or the current device if `use_rank` is False.
    If MPS is available but not CUDA, this device is returned using the given index or if None the given rank.
    If neither is available, the CPU device is returned.

    Args:
        device_idx: index value of device to use for CUDA or MPS, if None this will be chosen to be the default rank
            if `use_rank` is True (or MPS is selected), otherwise in CUDA the current device is selected.
        use_rank: if True, use the current rank in CUDA to select a device if `device_idx` is None, if False (the
            default), use the current device.
        device: if not None, create a device from this value and return it, all other arguments are ignored. This must
            be a device object already or a valid device string for constructing one.

    Returns:
        torch.device object for the selected accelerator device (or CPU if none is found), or the given device.

    Raises:
        ValueError: if `device_idx` is given and not an int, or outside the range [0, torch.cuda.device_count() - 1].
    """

    if device is not None:
        return torch.device(device)

    device_str = "cpu"

    if torch.cuda.is_available():
        if device_idx is not None:
            if not isinstance(device_idx, int) or device_idx < 0 or device_idx >= torch.cuda.device_count():
                raise ValueError(f"Invalid argument: {device_idx=}, device count is {torch.cuda.device_count()}.")
        elif use_rank:
            device_idx = get_default_rank()
        else:
            device_idx = torch.cuda.current_device()

        device_str = f"cuda:{int(device_idx)}"

    elif torch.backends.mps.is_available():
        if device_idx is not None:
            if device_idx < 0 or device_idx >= torch.mps.device_count():
                raise ValueError(f"Invalid argument: {device_idx=}, device count is {torch.mps.device_count()}.")
        else:  # no current_device for MPS so use rank regardless of `use_rank`
            device_idx = get_default_rank()

        device_str = f"mps:{int(device_idx)}"

    return torch.device(device_str)


def get_default_rank() -> int:
    """
    Returns the current rank if `torch.distributed` is initialized, else 0. This will succeed if distributed is not
    initialized so can be used to choose a default rank or device index in non-distributed environments.

    Returns:
        0 if torch.distributed is not initialized, otherwise the current rank (which may be 0).
    """
    return dist.get_rank() if dist.is_initialized() else 0


def is_rank_0() -> bool:
    """
    Returns True if the current rank is 0 or if not in a distributed environment, False otherwise. This will succeed in
    non-distributed environments with a return value of True, this can be used as a safe way of determining when to do
    things on rank 0 or when not using multiprocessing. For example, use this to disable logging on other ranks.

    Returns:
        True if the current rank is 0 or torch.distributed isn't initialized, False otherwise.
    """
    return get_default_rank() == 0


def find_device(obj: torch.tensor | torch.nn.Module) -> torch.device:
    """
    Returns the device of `obj` if it's a tensor, or searches for a tensor if a Module and returns its device.

    Args:
        obj: torch.tensor or torch.nn.Module to find a device at least part of it resides on.

    Returns:
        The found device.
    """
    if isinstance(obj, torch.nn.Module):
        return next(chain(obj.parameters(), obj.buffers()), torch.empty(0)).device
    else:
        return obj.device


def is_on_cpu(obj: torch.device | torch.tensor | torch.nn.Module) -> bool:
    """
    Returns True if the given device is the CPU device, or the given tensor or model is allocated in CPU memory.

    Args:
        obj: torch.device, torch.tensor, or torch.nn.Module object to check.

    Returns:
        True if the device is the CPU device, False otherwise.
    """
    if not isinstance(obj, torch.device):
        obj = find_device(obj)

    return obj.type == "cpu"


def get_dist_device(use_rank: bool = False) -> torch.device | None:
    """
    Returns the expected target device in the native PyTorch distributed data parallel environment. For NCCL backend,
    and if CUDA is available, this returns the CUDA device of current process (or current rank if `use_rank` is True).
    For GLOO backend, this returns the CPU device. For other cases, returns None as the default, `tensor.to(None)`
    will not change the device.

    Args:
        use_rank: if True, use the current rank to select a device, if False (the default), use the current device.

    Returns:
        torch.device object representing the selected CUDA device if using NCCL, CPU if using gloo, None otherwise.
    """
    backend = dist.get_backend() if dist.is_initialized() else None

    if backend == "nccl" and torch.cuda.is_available():
        return get_default_device(use_rank=use_rank)
    if backend == "gloo":
        return torch.device("cpu")

    return None  # dist not initialized or neither NCCL or gloo backends


def count_env_devices(visible_devices: str | None = None) -> int:
    """
    Count the devices given by the environment variable CUDA_VISIBLE_DEVICES or the supplied specified string if not
    None. This will validate the format, raising ValueError if the formatting is incorrect. This count is what is given
    and may be greater than the number of available devices.

    Args:
        visible_devices: device specified string, if None (the default) CUDA_VISIBLE_DEVICES is used instead.

    Returns:
        Number of specified devices

    Raises:
        ValueError: if any component of the specified string isn't a valid device index, UUID, or MIG specifier.
    """
    if visible_devices is None:
        visible_devices = os.environ.get(CUDA_VISIBLE_DEVICES, "").strip()

    if visible_devices in {"", "-1"}:
        return 0

    split_devices = [d.strip() for d in visible_devices.split(",")]

    if not all(split_devices):
        raise ValueError(f"Invalid device specifier with consecutive commas: {visible_devices}")

    index_pattern = r"\d+"
    uuid_pattern = r"(?:GPU|MIG)-[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    mig_spec_uuid = r"MIG-GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4})*(?:-[0-9a-fA-F]{12})?/\d+/\d+"
    item_regex = re.compile(f"^(?:{index_pattern}|{uuid_pattern}|{mig_spec_uuid})$")

    bad_devices = [d for d in split_devices if not item_regex.match(d)]

    if bad_devices:
        raise ValueError(f"Invalid device specifier with invalid devices: {bad_devices}")

    return len(split_devices)
