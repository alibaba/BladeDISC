load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

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
    includes = ["../../../"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@local_org_torch//:libtorch",
        "@org_mhlo_builder//csrc:mhlo_library",
    ],
    alwayslink = True,
)

cc_test(
    name = "mhlo_converter_register_test",
    srcs = ["mhlo_converter_register_test.cpp"],
    linkopts = [
        "-lpthread",
        "-lm",
        "-ldl",
    ],
    linkstatic = True,
    deps = [
        "//src/compiler/jit:torch_blade_jit",
        "//src/compiler/mlir/converters:torch_blade_mhlo_converter",
        "@googltest//:gtest_main",
        "@local_org_torch//:libtorch",
    ],
)
