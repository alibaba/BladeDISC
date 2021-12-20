#!/bin/env python

import os
import tensorflow as tf

TAO_OP_NAME = 'libtao_ops.so'
DISC_COMPILER_NAME = 'tao_compiler_main'
_ROOT = os.path.abspath(os.path.dirname(__file__))


def enable():
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
    tf.load_op_library(tao_op_path)
    print("Welcome BladeDISC!")
