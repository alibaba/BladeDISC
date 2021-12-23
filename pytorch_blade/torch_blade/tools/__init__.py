import contextlib
import torch
from .._torch_blade._tools import *
from .._torch_blade._tools import _jit_pass_onnx
from .._torch_blade._tools import _jit_pass_lower_simple_tuples
from .._torch_blade._tools import _jit_pass_const_loop_unrolling


@contextlib.contextmanager
def trust_tracing_shape(flag=True):
    old_flag = set_trust_tracing_shape(flag)
    try:
        yield
    finally:
        set_trust_tracing_shape(old_flag)

@contextlib.contextmanager
def record_cluster_io_context(flag=True):
    old_flag = set_record_cluster_io_flag(flag)
    try:
        yield
    finally:
        set_record_cluster_io_flag(old_flag)
