package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

filegroup(
    name = "torch_blade_jit_py_srcs",
    srcs = [
        "//pytorch_blade/compiler/jit/torch:onnx.cpp",
	"//pytorch_blade/compiler/jit/torch:pybind_utils.cpp",
    ] + glob([
        "**/pybind*.cpp",
    ]),
)

filegroup(
    name = "torch_blade_jit_py_hdrs",
    srcs = [
        "//pytorch_blade/compiler/jit/torch:onnx.h",
    ] + glob(["**/pybind*.h"]),
)

cc_library(
    name = "onnx_funcs",
    srcs = ["onnx_funcs.cpp"],
    hdrs = ["onnx_funcs.h"],
    deps = [
        "//pytorch_blade/common_utils:torch_blade_common",
        "@local_org_torch//:libtorch",
    ],
    alwayslink = True,
)

cc_library(
    name = "aten_custom_ops",
    srcs = ["aten_custom_ops.cpp"],
    deps = [
        "@local_org_torch//:libtorch",
    ],
    alwayslink = True,
)

cc_library(
    name = "tool_funcs",
    srcs = ["tool_funcs.cpp"],
    hdrs = ["tool_funcs.h"],
    deps = [
        "@local_org_torch//:libtorch",
    ],
)

cc_library(
    name = "torch_blade_jit",
    srcs = ["fusion.cpp", "shape_type_spec.cpp"],
    hdrs = [
        "fusion.h", "shape_type_spec.h"
    ],
    deps = [
        ":tool_funcs",
        "//pytorch_blade/compiler/jit/torch:shape_analysis",
        "//pytorch_blade/compiler/jit/torch:freeze_module",
        "//pytorch_blade/compiler/jit/torch:constant_propagation",
        "//pytorch_blade/common_utils:torch_blade_common",
        "@local_org_torch//:libtorch",
    ],
    alwayslink = True,
)

filegroup(
    name = "torch_blade_jit_test_srcs",
    srcs = glob(["*_test.cpp", "torch/constant_propagation_test.cpp"]),
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
        ":aten_custom_ops",
        ":tool_funcs",
        ":torch_blade_jit",
        "//pytorch_blade/common_utils:torch_blade_common",
        "@googltest//:gtest_main",
        "@local_org_torch//:libtorch",
    ]
)
