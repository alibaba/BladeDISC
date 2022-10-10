load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//bazel:build_defs.bzl", "if_platform_alibaba")

filegroup(
    name = "torch_blade_mhlo_converter_srcs",
    srcs = glob(
        ["*.cpp"],
        exclude = ["*_test.cpp"],
    ),
)

filegroup(
    name = "torch_blade_mhlo_converter_hdrs",
    srcs = glob(["*.h"]),
)

cc_library(
    name = "torch_blade_mhlo_converter",
    srcs = [":torch_blade_mhlo_converter_srcs"],
    hdrs = [":torch_blade_mhlo_converter_hdrs"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//pytorch_blade/common_utils:torch_blade_logging",
        "//pytorch_blade/common_utils:torch_blade_utils",
        "//pytorch_blade/compiler/jit:tool_funcs",
        "//pytorch_blade/compiler/jit:torch_blade_jit",
        "//pytorch_blade/torch-mlir:DiscTorchMLIRTorchToMhlo",
        "@local_org_torch//:libtorch",
        "@org_mhlo_builder//csrc:mhlo_library",
        "@torch-mlir-importer//:jit_ir_importer",
        "@llvm-project//mlir:CAPIIR",
    ],
    alwayslink = True,
)
