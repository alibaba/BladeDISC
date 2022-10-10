package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "torch_blade_mlir",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//pytorch_blade/common_utils:torch_blade_common",
        "//pytorch_blade/compiler/jit:torch_blade_jit",
        "//pytorch_blade/compiler/mlir/converters:torch_blade_mhlo_converter",
        "//pytorch_blade/compiler/mlir/converters/impl:torch_blade_mhlo_converter_impl",
        "//pytorch_blade/compiler/mlir/runtime:torch_blade_disc_rt",
        "@local_org_torch//:libtorch",
    ],
)

filegroup(
    name = "torch_blade_mlir_py_srcs",
    srcs = ["pybind_functions.cpp"],
)

filegroup(
    name = "torch_blade_mlir_py_hdrs",
    srcs = ["pybind_functions.h"],
)
