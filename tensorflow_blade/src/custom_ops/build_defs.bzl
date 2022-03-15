load("@local_config_tf//:build_defs.bzl", "tf_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load("@rules_python//python:defs.bzl", "py_test")

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

def tf_blade_ops_py_tests(srcs, deps = [], data = [], tags = None):
    """Create a py_test for each .py file in srcs."""
    all_deps = deps + ["//tests:tf_blade_ops_ut_common"]
    for file in srcs:
        if not file.endswith(".py"):
            fail("Need .py file in srcs, but got: " + file)
        name = file.rsplit(".", 1)[0]
        py_test(
            name = name,
            srcs = [file],
            deps = all_deps,
            data = data,
            tags = tags,
        )
