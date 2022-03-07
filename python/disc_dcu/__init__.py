from numpy import isin
from .disc_logging import logger
import os
import shutil
import tempfile
from .common import *

def PrepareEnv():
    path = os.path.dirname(os.path.abspath(__file__))
    rocm_path = os.environ.get("ROCM_PATH", None)
    if rocm_path is not None:
        os.environ["DISC_ROCM_PATH"] = rocm_path
    os.environ["TAO_USE_OPT_KERNEL"] = "1"
    os.environ["TAO_OPT_KERNEL_PATTERN_ROCM"] = os.path.join(path, "kernel_cache")
    os.environ["BRIDGE_ENABLE_TAO"] = "true"
    os.environ["TAO_COMPILER_PATH"] = "{}/tao_compiler_main".format(path)
    os.environ["TAO_ENABLE_CHECK"] = "false"
    os.environ["TAO_ENABLE_FALLBACK"] = "false"
    os.environ["TAO_COMPILATION_MODE_ASYNC"] = "false"
    os.environ["TF_XLA_FLAGS"]="--tf_xla_min_cluster_size=10"
    os.environ["TAO_ENABLE_MLIR"]="true"
    os.environ["TAO_MLIR_BRANCH_ONLY"]="true"
    os.environ["TAO_EXPERIMENTAL_ENABLE_MLIR_WHOLE_GRAPH_COMPILATION"]="true"
    os.environ["TAO_DUMP_PASS_OUTPUT"] = "false"
    # os.environ["DISC_PROFILING_CACHE"] = os.path.join(path, "profile_cache")
    # os.environ["DISC_KERNEL_CODEGEN_CACHE"] = os.path.join(path, "tmp_cache")
    import tensorflow.compat.v1 as tf
    tf.get_logger().setLevel('ERROR')
    # from deepmd.env import tf
    tf.load_op_library("{}/libtao_ops.so".format(path))

PrepareEnv()

from .kernel_opt import optimize_kernel
from enum import Enum

class OptLevel(Enum):
    O0 = 1
    O1 = 2
    O2 = 3

class DiscHandler(object):
    def __init__(self, optlevel=OptLevel.O0, limit=50, degree=[50, 100, 100], profiling_iter=50) -> None:
        path = os.path.dirname(os.path.abspath(__file__))
        self.cache = tempfile.mkdtemp()
        logger.debug("Temp cache {}".format(self.cache))
        self.cnt = 0
        self.opt = optlevel
        self.profiling_iter = profiling_iter
        self.limit = limit
        if not isinstance(degree, list):
            degree = [degree]
        self.degree = degree
        os.environ["DISC_PROFILING_CACHE"] = os.path.join(self.cache, "profile_cache")
        os.environ["DISC_KERNEL_CODEGEN_CACHE"] = os.path.join(self.cache, "tmp_cache")
        if not os.path.exists(os.path.join(self.cache, "profile_cache")):
            os.makedirs(os.path.join(self.cache, "profile_cache"))

    def start_profiling(self):
        os.environ["DISC_KERNEL_PROFILING"]="1"
    
    def finish_profiling(self):
        os.environ["DISC_KERNEL_PROFILING"]="0"

    def __enter__(self):
        if self.opt != OptLevel.O0:
            if self.cnt < self.profiling_iter: 
                self.start_profiling()

    def __exit__(self, type, value, traceback):
        if self.opt != OptLevel.O0:
            self.cnt+=1
            if self.cnt == self.profiling_iter:
                self.finish_profiling()
                optimize_kernel(self.opt == OptLevel.O2, cache=self.cache, limit=self.limit, degree=self.degree)
                shutil.rmtree(self.cache)
            
