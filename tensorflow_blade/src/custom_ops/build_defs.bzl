load("@local_config_tf//:build_defs.bzl", "tf_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")

def _cuda_copts(opts = []):
    """Gets the appropriate set of copts for (maybe) CUDA compilation.

    If we're doing CUDA compilation, returns copts for our particular CUDA
    compiler.  If we're not doing CUDA compilation, returns an empty list.

    """
    return select({
        "//conditions:default": [],
        "@local_config_cuda//cuda:using_nvcc": [
            "-nvcc_options=relaxed-constexpr",
            "-nvcc_options=ftz=true",
        ] + opts,
        "@local_config_cuda//cuda:using_clang": [
            "-fcuda-flush-denormals-to-zero",
        ] + opts,
    }) + if_cuda(["-DGOOGLE_CUDA=1"])

def tf_blade_cuda_library(deps = [], copts = [], cuda_copts = [], alwayslink = None, **kwargs):
    """create a `cc_library` that depends on CUDA and TF."""
    copts = copts + _cuda_copts(cuda_copts) + tf_copts()
    deps = deps + [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cub_headers",
    ]
    if alwayslink == None:
        alwayslink = True
    cuda_library(deps = deps, copts = copts, alwayslink = True, **kwargs)

def tf_blade_library(name, deps = [], alwayslink = None, copts = [], **kwargs):
    """create a `cc_library` that depends on TF."""
    copts = copts + tf_copts()
    deps = deps + [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ]
    if alwayslink == None:
        alwayslink = True
    native.cc_library(name = name, deps = deps, alwayslink = alwayslink, copts = copts, **kwargs)


