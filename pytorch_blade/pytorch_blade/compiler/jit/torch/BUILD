package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load(
    "//bazel:build_defs.bzl",
    "if_quantization_enabled"
)

cc_library(
    name = "shape_analysis",
    srcs = ["shape_analysis.cpp",
	    "operator_set.cpp",
	    "op_registry.cpp",
    ],
    hdrs = ["shape_analysis.h",
	    "operator_set.h",
	    "op_registry.h",
    ],
    deps = [
	"//pytorch_blade/common_utils:torch_blade_macros",
        "@local_org_torch//:libtorch",
    ] + if_quantization_enabled([
        "//pytorch_blade/quantization:quantization_op",
    ]),
    alwayslink = True,
)

cc_library(
    name = "freeze_module",
    srcs = ["alias_analysis.cpp",
            "const_loop_unroll.cpp",
	    "eliminate_redundant_permutations.cpp",
	    "freeze_module.cpp",
    ],
    hdrs = [
	    "alias_analysis.h",
	    "const_loop_unroll.h",
            "eliminate_redundant_permutations.h",
	    "freeze_module.h",
    ],
    deps = [
        "@local_org_torch//:libtorch",
    ],
    alwayslink = True,
)

cc_test(
    name = "shape_analysis_test",
    srcs = [
        "shape_analysis_test.cpp",
    ],
    linkopts = [
        "-lpthread",
        "-lm",
        "-ldl",
    ],
    linkstatic = True,
    deps = [
        ":shape_analysis",
        "@googltest//:gtest_main",
        "@local_org_torch//:libtorch",
    ]
)

exports_files([
    "onnx.h",
    "onnx.cpp",
    "pybind_utils.cpp",
])
