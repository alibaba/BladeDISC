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


import os
import shutil
import logging
from enum import Enum
from torch_blade.tools import read_bool_from_env
from torch_blade._torch_blade import _backends
from torch_blade.logging import logger

HASH_SEED = 1
DEFAULT_DISC_CACHE_DIR = os.path.join(os.path.expanduser('~'), ".cache/disc")

def enable_compilation_cache():
    return read_bool_from_env('TORCH_BLADE_ENABLE_COMPILATION_CACHE', False)

class ResultEnum(str, Enum):
    SO_BYTES = "so_bytes"
    PB_BYTES = "pb_bytes"
    INPUTS_DEVICE = "inputs_device"
    OUTPUTS_DEVICE = "outputs_device"

class CompilationResult:
    def __init__(self, so_bytes, pb_bytes, inputs_device, outputs_device):
        self._so_bytes = so_bytes
        self._pb_bytes = pb_bytes
        self._inputs_device = inputs_device
        self._outputs_device = outputs_device

    def _dump_data_to_file(self, path, data):
        if os.path.exists(path):
            raise RuntimeError("path {} does already exist.".format(path))
        if isinstance(data, str):
            data = bytes(data, "utf-8")
        with open(path, "wb") as f:
            f.write(data)

    def unpack(self):
        return self._so_bytes, self._pb_bytes, self._inputs_device, self._outputs_device

    @staticmethod
    def read_from_file(path):
        with open(path, "rb") as f:
            return f.read()

    def dump(self, path):
        self._dump_data_to_file(os.path.join(path, ResultEnum.SO_BYTES), self._so_bytes)
        self._dump_data_to_file(os.path.join(path, ResultEnum.PB_BYTES), self._pb_bytes)
        self._dump_data_to_file(os.path.join(path, ResultEnum.INPUTS_DEVICE), self._inputs_device)
        self._dump_data_to_file(os.path.join(path, ResultEnum.OUTPUTS_DEVICE), self._outputs_device)
        return True

    @staticmethod
    def load_from_path(path):
        so_bytes = CompilationResult.read_from_file(os.path.join(path, ResultEnum.SO_BYTES))
        pb_bytes = CompilationResult.read_from_file(os.path.join(path, ResultEnum.PB_BYTES))
        inputs_device = CompilationResult.read_from_file(os.path.join(path, ResultEnum.INPUTS_DEVICE))
        outputs_device = CompilationResult.read_from_file(os.path.join(path, ResultEnum.OUTPUTS_DEVICE))
        return CompilationResult(so_bytes, pb_bytes, inputs_device, outputs_device)

class DiscCompilationCache:
    def __init__(self, cache_dir=DEFAULT_DISC_CACHE_DIR):
        logger.info("DiscCompilationCache init with cache_dir: {}".format(cache_dir))
        self._cache_dir = cache_dir
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)
        

    def get(self, key) -> CompilationResult:
        path = os.path.join(self._cache_dir, str(key))
        if not os.path.exists(path):
            return None
        return CompilationResult.load_from_path(path)

    def has(self, key) -> bool:
        path = os.path.join(self._cache_dir, str(key))
        return os.path.exists(path)
    
    def set(self, hash_value : str, result : CompilationResult) -> bool:
        path = os.path.join(self._cache_dir, str(hash_value))
        if os.path.exists(path):
            shutil.rmtree(path)

        if not os.path.exists(path):
            os.makedirs(path)
        return result.dump(path)
