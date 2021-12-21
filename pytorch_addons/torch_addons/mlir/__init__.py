try:
    from .._torch_addons._mlir import *
    _is_available = True
except ImportError as e:
    # MLIR support is disable
    _is_available = False

import contextlib
from torch_addons import utils
from torch_addons.config import OptPipelines
from torch_addons.mlir.disc_engine_conversion import (
    _optimize_mlir,
    _compile_torchscript,
)

_DISC_GROUP_NAME = "disc_grp"
_DISC_TESTING_CONTEXT = False
OptPipelines.register_pipeline("DISC", _optimize_mlir)

def is_available():
    return _is_available

@contextlib.contextmanager
def testing_context():
    global _DISC_TESTING_CONTEXT
    old_mlir_testing_context = _DISC_TESTING_CONTEXT
    try:
        _DISC_TESTING_CONTEXT = True
        yield
    finally:
        _DISC_TESTING_CONTEXT = old_mlir_testing_context

def collect_engines(script_module):
    """
    Collect all engines in a script_module of disc
    """
    return utils.collect_engines(script_module, _DISC_GROUP_NAME)


def num_engines(script_module):
    """
    Return the number of engines of MLIR
    """
    return utils.num_engines(script_module, _DISC_GROUP_NAME)
