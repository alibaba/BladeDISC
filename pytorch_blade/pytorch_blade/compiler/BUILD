package(default_visibility = ["//visibility:public"])
load("//bazel:build_defs.bzl", "if_tensorrt_enabled")

filegroup(
    name = "torch_blade_py_srcs",
    srcs = [
        "//pytorch_blade/compiler/jit:torch_blade_jit_py_srcs",
        "//pytorch_blade/compiler/mlir:torch_blade_mlir_py_srcs",
    ] + if_tensorrt_enabled(["//pytorch_blade/compiler/tensorrt:pybind_functions.cpp"]),
)

filegroup(
    name = "torch_blade_py_hdrs",
    srcs = [
        "//pytorch_blade/compiler/jit:torch_blade_jit_py_hdrs",
        "//pytorch_blade/compiler/mlir:torch_blade_mlir_py_hdrs",
    ] + if_tensorrt_enabled(["//pytorch_blade/compiler/tensorrt:pybind_functions.h"]),
)
