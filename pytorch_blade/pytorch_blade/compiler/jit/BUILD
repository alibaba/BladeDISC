package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

filegroup(
    name = "torch_blade_jit_srcs",
    srcs = glob(
        ["**/*.cpp"],
        exclude = [
            "**/torch/onnx.cpp",
            "**/*test.cpp",
            "**/pybind*.cpp",
        ],
    ),
)

filegroup(
    name = "torch_blade_jit_hdrs",
    srcs = glob(
        ["**/*.h"],
        exclude = ["**/pybind*.h"],
    ),
)

cc_library(
    name = "torch_blade_jit",
    srcs = [":torch_blade_jit_srcs"],
    hdrs = [
        ":torch_blade_jit_hdrs",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//pytorch_blade/common_utils:torch_blade_common",
        "@local_org_torch//:libtorch",
    ],
    alwayslink = True,
)

filegroup(
    name = "torch_blade_jit_py_srcs",
    srcs = glob([
        "**/torch/onnx.cpp",
        "**/pybind*.cpp",
    ]),
)

filegroup(
    name = "torch_blade_jit_py_hdrs",
    srcs = glob(["**/pybind*.h"]),
)

cc_library(
    name = "tool_funcs",
    srcs = ["tool_funcs.cpp"],
    hdrs = ["tool_funcs.h"],
    deps = [
        "@local_org_torch//:libtorch",
    ],
)

filegroup(
    name = "torch_blade_jit_test_srcs",
    srcs = glob(["**/*_test.cpp"]),
)

cc_test(
    name = "jit_test",
    srcs = [
        ":torch_blade_jit_test_srcs",
    ],
    linkopts = [
        "-lpthread",
        "-lm",
        "-ldl",
    ],
    linkstatic = True,
    deps = [
        ":tool_funcs",
        ":torch_blade_jit",
        "//pytorch_blade/common_utils:torch_blade_common",
        "@googltest//:gtest_main",
        "@local_org_torch//:libtorch",
    ],
)
