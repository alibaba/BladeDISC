import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


TAO_OP_NAME = 'libtao_ops.so'
DISC_COMPILER_NAME = 'tao_compiler_main'
_ROOT = os.path.abspath(os.path.dirname(__file__))
_ROOT = "/global/home/aliliang/aicompiler/bladedisc/workspace/venv/lib/python3.6/site-packages/disc_dcu/"
_ROOT = "/home/fl237079/workspace/tao_built"

"""
def enable(disc_cpu=False, num_intra_op_threads=1, fast_math_level=4):
    tao_op_path = os.path.join(_ROOT, TAO_OP_NAME)
    disc_compiler_path = os.path.join(_ROOT, DISC_COMPILER_NAME)
    os.environ.setdefault("BRIDGE_ENABLE_TAO", "true")
    os.environ.setdefault("TAO_COMPILER_PATH", disc_compiler_path)
    os.environ.setdefault("TAO_COMPILATION_MODE_ASYNC", "false")
    os.environ.setdefault("TAO_MLIR_BRANCH_ONLY", "true")
    os.environ.setdefault("OMP_NUM_THREADS", str(num_intra_op_threads))
    os.environ.setdefault("DISC_CPU_FAST_MATH_LEVEL", str(fast_math_level))
    tf.load_op_library(tao_op_path)
    print("Welcome BladeDISC!")"""

def enable(disc_cpu=False, num_intra_op_threads=1, fast_math_level=4):
    tao_op_path = os.path.join(_ROOT, TAO_OP_NAME)
    disc_compiler_path = os.path.join(_ROOT, DISC_COMPILER_NAME)
    os.environ.setdefault("BRIDGE_ENABLE_TAO", "true")
    os.environ.setdefault("TAO_COMPILER_PATH", disc_compiler_path)
    os.environ.setdefault("TAO_COMPILATION_MODE_ASYNC", "false")
    os.environ["TAO_ENABLE_MLIR"]="true"
    os.environ.setdefault("TAO_MLIR_BRANCH_ONLY", "true")
    os.environ["TF_XLA_FLAGS"]="--tf_xla_min_cluster_size=10"
    os.environ["TAO_EXPERIMENTAL_ENABLE_MLIR_WHOLE_GRAPH_COMPILATION"]="true"
    #os.environ.setdefault("OMP_NUM_THREADS", str(num_intra_op_threads))
    os.environ.setdefault("DISC_CPU_FAST_MATH_LEVEL", str(fast_math_level))
    tf.load_op_library(tao_op_path)
    print("Welcome BladeDISC!")

enable()
