package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "torch_blade_mlir",
    deps = [
        "//src/common_utils:torch_blade_common",
        "//src/compiler/jit:torch_blade_jit",
        "//src/compiler/mlir/converters:torch_blade_mhlo_converter",
        "//src/compiler/mlir/converters/impl:torch_blade_mhlo_converter_impl",
        "//src/compiler/mlir/runtime:torch_blade_disc_rt",
        "@local_org_torch//:ATen",
        "@local_org_torch//:libtorch",
    ],
    alwayslink = True,
    visibility = [
        "//visibility:public",
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
