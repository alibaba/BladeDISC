# Copyright 2021 The BladeDISC Authors. All rights reserved.
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
import subprocess
import tempfile
from datetime import datetime
from io import BytesIO

import torch
import torch_blade
from torch_blade import mlir, tools, utils
from torch_blade._torch_blade import _backends
from torch_blade.clustering import support_fusion_group, support_group_conversion
from torch_blade.config import Config
from torch_blade.logging import logger
from torch_blade.mlir.cache import DiscCompilationCache, CompilationResult, enable_compilation_cache
from torch_blade.mlir.hash import get_graph_hash
from collections import defaultdict

def _dump_to_tempfile(tmp_dir, dump_bytes):
    inp_file = tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False)
    inp_file.write(bytes(dump_bytes, "utf-8"))
    inp_file.close()
    return inp_file

disc_cache = None
if enable_compilation_cache():
    disc_cache = DiscCompilationCache()

def _compile_torchscript(graph):
    # NB: Some MLIR debug information would be dump to mlir_dump_dir,
    # since the feature is not stable currently.
    # (Only files of the last run are reserved except for compile log.)
    mlir_dump_dir = "dump_dir"
    pkg_path = os.path.dirname(os.path.abspath(torch_blade.__file__))
    mhlo_bytes, pretty_bytes, input_dev_str, output_dev_str = mlir.cvt_torchscript_to_mhlo(graph)
    mhlo_compile_cmd = os.path.join(pkg_path, "disc_compiler_main")
    debug_log_enabled = tools.read_bool_from_env('TORCH_BLADE_DEBUG_LOG', False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        time_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S.%f')
        # dump the parsable/compilable mlir bytes into file
        inp_mlir_file = _dump_to_tempfile(tmp_dir, mhlo_bytes)
        # dump the pretty mlir bytes(for debug) into file
        mlir_pretty_file = _dump_to_tempfile(tmp_dir, pretty_bytes)

        if not debug_log_enabled:
            mlir_dump_dir = os.path.join(tmp_dir, mlir_dump_dir)
        # copy mlir files to mlir_dump_dir
        if not os.path.exists(mlir_dump_dir):
            os.makedirs(mlir_dump_dir)
        assert os.path.isdir(mlir_dump_dir), "the dump mlir path must be a directory"
        with open(os.path.join(mlir_dump_dir, f'graph.{time_str}.txt'), 'w') as f:
            f.write(str(graph))
        # the mhlo_compiler output binary file
        out_file_name = tempfile.NamedTemporaryFile(
            dir=tmp_dir, suffix=".so", delete=False
        ).name

        # the mhlo_compiler output metadata file
        out_file_pbtxt = out_file_name + ".pbtxt"
        compile_log = os.devnull
        env = os.environ.copy()
        if debug_log_enabled:
            env['TF_CPP_VMODULE'] = "disc_compiler=1"
            compile_log = os.path.join(mlir_dump_dir, "mhlo_compile." + time_str + ".log")
            shutil.copy(inp_mlir_file.name, os.path.join(mlir_dump_dir, f"dump.{time_str}.mlir"))
            shutil.copy(
                mlir_pretty_file.name, os.path.join(mlir_dump_dir, f"dump.{time_str}.pretty.mlir")
            )

        with open(compile_log, "w") as devnull:
            cfg = Config.get_current_context_or_new()
            env['TAO_MLIR_ENABLE_AMP'] = str(cfg.enable_mlir_amp).lower()
            env['DISC_CPU_FAST_MATH_LEVEL'] = str(cfg.disc_cpu_fast_math_level)
            # RUN: disc_compiler_main input_mlir_file.mlir output_file.so
            # redirect stdout to devnull
            extra_flags = []
            if cfg.disc_compile_for_multi_cuda_targets:
                extra_flags += ["--multi-cc-support"]
            
            subprocess.check_call(
                [mhlo_compile_cmd, inp_mlir_file.name, out_file_name, "--mlir-elide-elementsattrs-if-larger=8"] + extra_flags,
                stdout=devnull,
                stderr=devnull,
                env=env,
            )
        assert os.path.exists(out_file_name)
        assert os.path.exists(out_file_pbtxt)
        with open(out_file_name, "rb") as f:
            so_bytes = f.read()
        with open(out_file_pbtxt, "rb") as f_pbtxt:
            pb_bytes = f_pbtxt.read()

        if debug_log_enabled:
            # copy result to mlir_dump_dir
            shutil.move(out_file_name, os.path.join(mlir_dump_dir, f"out.{time_str}.so"))
            shutil.move(out_file_pbtxt, os.path.join(mlir_dump_dir, f"out.{time_str}.so.pbtxt"))

        return so_bytes, pb_bytes, input_dev_str, output_dev_str

def _get_customize_op_black_list(graph):
    cfg = Config.get_current_context_or_new()
    node_kind_list = defaultdict(list)
    for node in graph.node_list():
        node_kind_list[node.kind()].append(node)

    customize_node_black_list = [
        op for op in cfg.customize_op_black_list if "@" in op
    ]
    extra_unspt_nodes = []
    for node_name in customize_node_black_list:
        try:
            op_kind, node_idx = node_name.split("@")
            node_idx = int(node_idx)
            node = node_kind_list[op_kind][node_idx]
            extra_unspt_nodes.append(node)
        except Exception as error:
            logger.warning(error)

    customize_node_black_list = [
        op for op in cfg.customize_op_black_list if "@" not in op
    ]
    extra_unspt_nodes += [n for n in graph.nodes() if n.kind() in customize_node_black_list]
    return extra_unspt_nodes

def _get_mlir_unsupported(graph):
    extra_unspt_nodes = _get_customize_op_black_list(graph)
    unspt_nodes = [n for n in graph.nodes() if not mlir.is_mlir_mhlo_supported(n)]
    return set(unspt_nodes + extra_unspt_nodes)

def _subgraph_to_bytes(subgraph, group_name):
    fallback_module = utils.subgraph_to_module(subgraph, group_name)
    m_bytes = BytesIO()
    torch.jit.save(fallback_module, m_bytes)
    return m_bytes.getvalue()

def _disc_engine_conversion(module):
    def try_cvt_to_disc_engine_func(
            c_module, c_module_lock, subgraph, group_name, grp_calib_data=None
    ):
        attr_name = f"{mlir._DISC_GROUP_NAME}{group_name}"
        try:
            graph_hash = get_graph_hash(subgraph)
            logger.info(f"graph hash: {graph_hash}")
            if disc_cache:
                if disc_cache.has(graph_hash):
                    logger.info(f"hit disc cache {graph_hash}")
                    result = disc_cache.get(graph_hash)
                    so_bytes, pb_bytes, input_dev_str, output_dev_str = result.unpack()
                else:
                    logger.info(f"compile torchscript for {graph_hash}")
                    so_bytes, pb_bytes, input_dev_str, output_dev_str = _compile_torchscript(subgraph)
                    logger.info(f"so_bytes: {len(so_bytes)}, pb_bytes: {len(pb_bytes)}, input_dev_str: {input_dev_str}, output_dev_str: {output_dev_str}")
                    logger.info(f"set disc cache {graph_hash}")
                    disc_cache.set(graph_hash, CompilationResult(so_bytes, pb_bytes, input_dev_str, output_dev_str))
            else:
                so_bytes, pb_bytes, input_dev_str, output_dev_str = _compile_torchscript(subgraph)

            state = _backends.EngineState()
            state.inputs = [_backends.TensorInfo(inp) for inp in subgraph.inputs()]
            state.outputs = [_backends.TensorInfo(out) for out in subgraph.outputs()]
            state.engine_bytes = so_bytes
            state.model_proto = pb_bytes
            state.backend_name = mlir.backend_name()
            debug_fallback = tools.read_bool_from_env('TORCH_BLADE_DEBUG_ENABLE_ERROR_FALLBACK', False)
            if debug_fallback:
                fallback_bytes = _subgraph_to_bytes(subgraph, attr_name)
            else:
                fallback_bytes = ""

            # register engine into module, something like:
            # __torch__.torch.classes.torch_blade.Engine = prim::GetAttr[name="disc_grp0"](%self)
            with c_module_lock:
                eng_type = _backends.register_engine(
                    c_module,
                    state,
                    attr_name,
                    fallback_bytes,
                    str(subgraph),
                )

            return attr_name, eng_type
        except Exception as error:
            logger.warning(error)
            return None

    support_group_conversion.group_to_engine_conversion(
        module, try_cvt_to_disc_engine_func, adapt_number_ios=True
    )


def _optimize_mlir(script_module):
    """
    Given a ScriptModule, do MLIR optimization on it.

    Args:
        script_module(torch.jit.ScriptModule): PyTorch ScriptModule to be optimized by MLIR.
    """
    assert isinstance(
        script_module, torch.jit.ScriptModule
    ), "Only torch.jit.ScriptModule can be optimized by TensorRT, but {} is given.".format(
        type(script_module)
    )
    # do tensorrt optimization
    c_module = script_module._c
    graph = c_module.forward.graph

    # NOTE: this is NOT SAFE, since it assumes that the LHS is not aliased by
    # another value. This is only to avoid breaking ONNX export; when alias
    # analysis is done we can emit a warning if someone tries to export.
    #
    # TODO: Replace it with a safe version, that only replace inplace with out-of-place ops,
    # only when it's safe. Otherwise label the inplace ops as unsupported.
    # Then move the pass as a common step to _optimize_common.
    torch._C._jit_pass_remove_inplace_ops(graph)
    def fusion_block(block):
        for n in block.nodes():
            for blk in n.blocks():
                fusion_block(blk)

        unsupported_nodes = _get_mlir_unsupported(block)

        _ = support_fusion_group.supported_node_fusion(
            graph, block, unsupported_nodes, support_number_ios=True
        )

    with tools.trust_tracing_shape():
        fusion_block(graph)
        _disc_engine_conversion(c_module)
