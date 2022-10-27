from numpy import isin
from .disc_logging import logger
import os
import shutil
import tempfile
from .common import *

from .kernel_opt import optimize_kernel
from enum import Enum

def prepare_env():
    path = os.path.dirname(os.path.abspath(__file__))
    rocm_path = os.environ.get("ROCM_PATH", None)
    if rocm_path is not None:
        os.environ["DISC_ROCM_PATH"] = rocm_path
    os.environ["BLADNN_KERNEL_CACHE"] = os.path.join(path, "kernel_cache")
    os.environ["BRIDGE_ENABLE_TAO"] = "true"
    os.environ["TAO_COMPILER_PATH"] = "{}/tao_compiler_main".format(path)
    os.environ["TAO_ENABLE_CHECK"] = "false"
    os.environ["TAO_ENABLE_FALLBACK"] = "false"
    os.environ["TAO_COMPILATION_MODE_ASYNC"] = "false"
    os.environ["TF_XLA_FLAGS"]="--tf_xla_min_cluster_size=1"
    os.environ["TAO_ENABLE_MLIR"]="true"
    os.environ["TAO_MLIR_BRANCH_ONLY"]="true"
    os.environ["TAO_EXPERIMENTAL_ENABLE_MLIR_WHOLE_GRAPH_COMPILATION"]="true"
    os.environ["TAO_DUMP_PASS_OUTPUT"] = "false"
    import tensorflow.compat.v1 as tf
    tf.get_logger().setLevel('ERROR')
    # from deepmd.env import tf
    tf.load_op_library("{}/libtao_ops.so".format(path))

class OptLevel(Enum):
    O0 = 1
    O1 = 2
    O2 = 3

class DiscOptItertionContext(object):
    """Used inside each iteration of the training process to automatically handle 
    the kernel collection, kernel tuning and kernel generation
    """

    def __init__(self, optlevel=OptLevel.O0, limit=50, degree=[50, 100, 100], profiling_iter=50) -> None:
        """
        Parameters
        ----------
        optlevel : OptLevel
            The level of optimization. O0 means no kernel tuning and no kernel implementation generation.
            O1 means no kernel tuning but to generate optimized kernel implementation from existing tuning records.
            O2 means to tune and generate kernel implementations.

        limit : int
            Only tune or generate optimized implementations for the top [limit] time-consuming kernels.

        degree : a list of 3 ints
            The first int means the tuning step for ansor tuning.
            The second int means the tuning step for autotvm tuning using no-transpose template.
            The third int means the tuning step for autotvm tuning using transpose template.
            An int no greater than zero means no tunnig for that case.

        profiling_iter : int
            The number of iterations to collect the kernels to be optimized.
            Targeting kernels will be collected after [profiling_iter] iterations to be optimized
        """
        prepare_env()
        self._cache = tempfile.mkdtemp()
        logger.debug("Temp cache {}".format(self._cache))
        self._cnt = 0
        self._opt = optlevel
        self._profiling_iter = profiling_iter
        self._limit = limit
        self._profile_cache = os.path.join(self._cache, "profile_cache")
        if not isinstance(degree, list):
            degree = [degree]
        self._degree = degree
        os.environ["BLADNN_COLLECT_CACHE"] = self._profile_cache
        os.environ["DISC_PROFILING_CACHE"] = self._profile_cache
        os.environ["BLADNN_CODEGEN_TMP"] = os.path.join(self._cache, "tmp_cache")
        if not os.path.exists(self._profile_cache):
            os.makedirs(self._profile_cache)

    def _start_profiling(self):
        os.environ["BLADNN_COLLECT_STATUS"]="1"
    
    def _finish_profiling(self):
        os.environ["BLADNN_COLLECT_STATUS"]="0"
    
    def _start_collecting(self):
        os.environ["BLADNN_COLLECT_STATUS"]="2"

    def __enter__(self):
        if self._opt != OptLevel.O0:
            if self._cnt < self._profiling_iter: 
                self._start_profiling()
            elif self._cnt == self._profiling_iter: 
                self._start_collecting()

    def __exit__(self, type, value, traceback):
        if self._opt != OptLevel.O0:
            self._cnt+=1
            if self._cnt == self._profiling_iter + 1:
                self._finish_profiling()
                optimize_kernel(self._opt == OptLevel.O2, cache=self._cache, limit=self._limit, degree=self._degree)
                shutil.rmtree(self._cache)