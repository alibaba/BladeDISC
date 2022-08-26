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
        "//src/common_utils:torch_blade_logging",
        "//src:torch_blade_include",
        "//src/torch-mlir:DiscTorchMLIRTorchToMhlo",
        "@local_org_torch//:libtorch",
        "@org_mhlo_builder//csrc:mhlo_library",
        "@torch-mlir-importer//:jit_ir_importer",
        "@llvm-project//mlir:CAPIIR",
    ],
    alwayslink = True,
)

cc_test(
    name = "mhlo_converter_register_test",
    srcs = [
        "mhlo_converter_register_test.cpp",
    ],
    linkopts = [
        "-lpthread",
        "-lm",
        "-ldl",
    ],
    linkstatic = True,
    deps = [
        "//src:torch_blade_include",
        "//src:torch_blade_shared",
        "@zlib",
        "@googltest//:gtest_main",
        "@local_org_torch//:libtorch",
        "@org_mhlo_builder//csrc:mhlo_library",
        "@org_tensorflow//tensorflow/compiler/mlir/xla/ral:ral_base_context_lib",
    ] + if_platform_alibaba([
        "//src/platform_alibaba:runtime_deps",
    ])
)
