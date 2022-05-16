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
        "//src:torch_blade_include",
        "//src/common_utils:torch_blade_logging",
	"//torch_xla/csrc/client:mhlo_builder",
	# "//torch_xla/csrc:torch_xla",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:constants",
	# "//torch_xla/csrc/client:xla_builder",
        "@local_org_torch//:libtorch",
        "@org_mhlo_builder//csrc:mhlo_library",
    ],
    alwayslink = True,
)

cc_test(
    name = "mhlo_converter_register_test",
    srcs = [
        "mhlo_converter_register_test.cpp",
        "//src:libtorch_blade.so",
    ],
    linkopts = [
        "-lpthread",
        "-lm",
        "-ldl",
    ],
    linkstatic = True,
    deps = [
        "//src:libtorch_blade.so",
        "@zlib",
        "@googltest//:gtest_main",
        "@local_org_torch//:libtorch",
        "@org_tensorflow//tensorflow/compiler/mlir/xla/ral:ral_base_context_lib",
    ],
)
