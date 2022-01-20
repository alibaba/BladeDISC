#!/bin/env python
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
import tensorflow as tf

TAO_OP_NAME = 'libtao_ops.so'
DISC_COMPILER_NAME = 'tao_compiler_main'
_ROOT = os.path.abspath(os.path.dirname(__file__))


def enable(disc_cpu=False, num_intra_op_threads=1, fast_math_level=4):
    '''Set up BladeDISC.

    This is a simple method to enable BladeDISC for TensorFlow.

    Parameters
    ----------
    disc_cpu : bool
        Enable cpu JIT compilation optimization if True.

    num_intra_op_threads: int
        The number of threads that BladeDISC uses for the execution of
        the compiled part of the graph. It's recommanded to use the same
        value as the TF_NUM_INTRAOP_THREADS.

    fast_math_level: int
        Controls the extent that BladeDISC is allowed to use fast math for
        acceleration. Higher number usually means faster speed while it may
        lead to some accuracy loss in some cases.
          Level 0: no fast math
          Level 1: apply approximation for some expensive math ops (e.g. exp, sin)
          Level 2: Level 1 + AllowReassoc
          Level 3: Level 2 + NoNaNs + NoSignedZeros
          Level 4: Level 3 + fully llvm fast math
    '''
    tao_op_path = os.path.join(_ROOT, TAO_OP_NAME)
    disc_compiler_path = os.path.join(_ROOT, DISC_COMPILER_NAME)
    if not os.path.exists(tao_op_path):
        raise FileNotFoundError(
            "can not find libtao_ops.so on {}".format(tao_op_path))
    if not os.path.exists(disc_compiler_path):
        raise FileNotFoundError(
            "can not find tao_compiler_main on {}".format(disc_compiler_path))
    os.environ.setdefault("BRIDGE_ENABLE_TAO", "true")
    os.environ.setdefault("TAO_COMPILER_PATH", disc_compiler_path)
    os.environ.setdefault("TAO_COMPILATION_MODE_ASYNC", "false")
    os.environ.setdefault("TAO_MLIR_BRANCH_ONLY", "true")
    if disc_cpu:
        os.environ.setdefault("TAO_FLAGS", "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit")
        os.environ.setdefault("OMP_NUM_THREADS", str(num_intra_op_threads))
        os.environ.setdefault("DISC_CPU_FAST_MATH_LEVEL", str(fast_math_level))
    tf.load_op_library(tao_op_path)
    print("Welcome BladeDISC!")
