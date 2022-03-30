load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_dcu",
    "if_rocm",
)

def disc_cc_library(copts = tf_copts(), **kwargs):
    """Generate a cc_library with device related copts.
    """
    cc_library(
        copts = (copts + if_cuda(["-DGOOGLE_CUDA=1"]) + if_dcu(["-DTENSORFLOW_USE_DCU=1"]) + if_rocm(["-DTENSORFLOW_USE_ROCM=1"])),
        **kwargs
    )

def if_platform_alibaba(if_true, if_false=[]):
    return select({
        "//tensorflow/compiler/mlir/disc:is_platform_alibaba": if_true,
        "//conditions:default": if_false
    })

def if_cuda_or_rocm(if_true, if_false=[]):
    return select({
        "@local_config_cuda//:is_cuda_enabled": if_true,
        "@local_config_rocm//rocm:using_hipcc": if_true,
        "//conditions:default": if_false
    })

def if_patine(if_true, if_false=[]):
    return select({
        "//tensorflow/compiler/mlir/disc:is_patine": if_true,
        "//conditions:default": if_false
    })

def if_blade_gemm(if_true, if_false=[]):
    return select({
        "//tensorflow/compiler/mlir/disc:is_blade_gemm": if_true,
        "//conditions:default": if_false
    })

def if_mkldnn(if_true, if_false=[]):
    return select({
        "//tensorflow/compiler/mlir/disc:is_mkldnn": if_true,
        "//conditions:default": if_false
    })
