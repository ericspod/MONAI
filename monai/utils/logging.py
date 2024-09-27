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

import logging
import sys

__all__ = ["get_logger"]

DEFAULT_FMT = "%(asctime)s - %(levelname)s - %(message)s"


def get_logger(
    module_name: str = "monai.utils",
    fmt: str = DEFAULT_FMT,
    datefmt: str | None = None,
    logger_handler: logging.Handler | None = None,
) -> logging.Logger:
    """
    Get a `module_name` logger with the specified format and date format.
    By default, the logger will print to `stdout` at the INFO level.
    If `module_name` is `None`, return the root logger.
    `fmt` and `datafmt` are passed to a `logging.Formatter` object
    (https://docs.python.org/3/library/logging.html#formatter-objects).
    `logger_handler` can be used to add an additional handler.
    """
    adds_stdout_handler = module_name is not None and module_name not in logging.root.manager.loggerDict
    logger = logging.getLogger(module_name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if adds_stdout_handler:  # don't add multiple stdout or add to the root
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if logger_handler is not None:
        logger.addHandler(logger_handler)
    return logger
