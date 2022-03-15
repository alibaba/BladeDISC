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
            "**/freeze_module.cpp" # TODO(yancey.yx): fix this pass to pass building with PyTorch LTC branch
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
        "//src/common_utils:torch_blade_common",
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

cc_test(
    name = "shape_type_test",
    srcs = [
        "shape_type_spec.cpp",
        "shape_type_test.cpp",
    ],
    linkopts = [
        "-lpthread",
        "-lm",
        "-ldl",
    ],
    linkstatic = True,
    deps = [
        ":tool_funcs",
        "//src/common_utils:torch_blade_common",
        "@googltest//:gtest_main",
        "@local_org_torch//:libtorch",
    ],
)
