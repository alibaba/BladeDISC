# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch


def parse_version(version: str) -> Tuple[int]:
    """
    Parses a version string into (major, minor, patch) version numbers.

    Args:
      version: Full version number string, possibly including revision / commit hash.

    Returns:
      An int 3-tuple of (major, minor, patch) version numbers.
    """
    # Extract version number part (i.e. toss any revision / hash parts).
    version_number_str = version
    for i in range(len(version)):
        c = version[i]
        if not (c.isdigit() or c == "."):
            version_number_str = version[:i]
            break

    return tuple([int(n) for n in version_number_str.split(".")[:3]])


def torch_version() -> Tuple[int]:
    return parse_version(torch.__version__)
