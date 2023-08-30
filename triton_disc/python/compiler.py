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
import json
import torch
import triton
from pathlib import Path
import triton._C.libtriton.triton as _triton
from triton.compiler import (
    CacheManager,
    CompiledKernel,
    parse_mlir_module,
    ast_to_ttir,
    ttir_to_ttgir,
    ttgir_to_llir,
    llir_to_ptx,
    ptx_to_cubin,
    convert_type_repr,
    make_hash,
    make_stub,
    ptx_get_kernel_name,
    instance_descriptor,
    prototype_pattern,
    arg_type_pattern
)


# def compile(fn, signature: str, device: int = -1, constants=dict(), num_warps: int = 4, num_stages: int = 3, extern_libs=None, configs=None):
def compile(fn, **kwargs):
    print("begin triton.compile...", flush=True)
    capability = kwargs.get("cc", None)
    if capability is None:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
    # we get the kernel, i.e. the first function generated in the module
    # if fn is not a JITFunction, then it
    # has to be a path to a file
    context = _triton.ir.context()
    asm = dict()
    constants = kwargs.get("constants", dict())
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 3 if capability >= 75 else 2)
    extern_libs = kwargs.get("extern_libs", dict())
    # build compilation stages
    stages = {
        "ast": (lambda path: fn, None),
        "ttir": (lambda path: parse_mlir_module(path, context),
                 lambda src: ast_to_ttir(src, signature, configs[0], constants)),
        "ttgir": (lambda path: parse_mlir_module(path, context),
                  lambda src: ttir_to_ttgir(src, num_warps, num_stages, capability)),
        "llir": (lambda path: Path(path).read_text(),
                 lambda src: ttgir_to_llir(src, extern_libs, capability)),
        "ptx": (lambda path: Path(path).read_text(),
                lambda src: llir_to_ptx(src, capability)),
        "cubin": (lambda path: Path(path).read_bytes(),
                  lambda src: ptx_to_cubin(src, capability))
    }
    # find out the signature of the function
    if isinstance(fn, triton.runtime.JITFunction):
        configs = kwargs.get("configs", None)
        signature = kwargs["signature"]
        if configs is None:
            configs = [instance_descriptor()]
        assert len(configs) == 1
        kwargs["configs"] = configs
        name = fn.__name__
        first_stage = 0
        if isinstance(signature, str):
            signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
        kwargs["signature"] = signature
    else:
        assert isinstance(fn, str)
        _, ir = os.path.basename(fn).split(".")
        src = Path(fn).read_text()
        import re
        match = re.search(prototype_pattern[ir], src, re.MULTILINE)
        name, signature = match.group(1), match.group(2)
        # print(name, signature)
        types = re.findall(arg_type_pattern[ir], signature)
        # print(types)
        param_tys = [convert_type_repr(ty) for ty in types]
        signature = {k: v for k, v in enumerate(param_tys)}
        first_stage = list(stages.keys()).index(ir)

    # cache manager
    so_path = make_stub(name, signature, constants)
    # create cache manager
    fn_cache_manager = CacheManager(make_hash(fn, **kwargs))
    # determine name and extension type of provided function
    if isinstance(fn, triton.runtime.JITFunction):
        name, ext = fn.__name__, "ast"
    else:
        name, ext = os.path.basename(fn).split(".")

    # load metadata if any
    metadata = None
    if fn_cache_manager.has_file(f'{name}.json'):
        with open(fn_cache_manager._make_path(f"{name}.json")) as f:
            metadata = json.load(f)
    else:
        metadata = {"num_warps": num_warps, "num_stages": num_stages, "ctime": dict()}
        if ext == "ptx":
            assert "shared" in kwargs, "ptx compilation must provide shared memory size"
            metadata["shared"] = kwargs["shared"]

    first_stage = list(stages.keys()).index(ext)
    asm = dict()
    module = fn
    # run compilation pipeline  and populate metadata
    for ir, (parse, compile) in list(stages.items())[first_stage:]:
        print(f"pipeline {ir}, {module}", flush=True)
        path = fn_cache_manager._make_path(f"{name}.{ir}")
        if ir == ext:
            next_module = parse(fn)
        elif os.path.exists(path) and\
                ir in metadata["ctime"] and\
                os.path.getctime(path) == metadata["ctime"][ir]:
            next_module = parse(path)
        else:
            next_module = compile(module)
            fn_cache_manager.put(next_module, f"{name}.{ir}")
        # print(ir, next_module, flush=True)
        if os.path.exists(path):
            metadata["ctime"][ir] = os.path.getctime(path)
        asm[ir] = next_module if ir == "cubin" else str(next_module)
        if ir == "llir" and "shared" not in metadata:
            metadata["shared"] = _triton.get_shared_memory_size(module)
        if ir == "ptx":
            metadata["name"] = ptx_get_kernel_name(next_module)
        module = next_module
    # write-back metadata
    fn_cache_manager.put(json.dumps(metadata), f"{name}.json", binary=False)
    # return handle to compiled kernel
    return CompiledKernel(so_path, metadata, asm)
