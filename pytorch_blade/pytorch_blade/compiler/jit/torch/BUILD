package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "shape_analysis",
    srcs = ["shape_analysis.cpp",
	    "schema_set.cpp",
	    "op_registry.cpp",
    ],
    hdrs = ["shape_analysis.h",
	    "schema_set.h",
	    "op_registry.h",
    ],
    deps = [
	"//pytorch_blade/common_utils:torch_blade_macros",
        "@local_org_torch//:libtorch",
    ],
    copts = select({
       "//:enable_cuda": ["-DTORCH_BLADE_BUILD_WITH_CUDA"],
       "//conditions:default": []}),
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
        "//pytorch_blade/common_utils:torch_blade_macros",
        "@local_org_torch//:libtorch",
    ],
    alwayslink = True,
)

cc_library(
    name = "constant_propagation",
    srcs = [
        "constant_propagation.cpp",
    ],
    hdrs = [
        "alias_analysis.h",
	    "constant_propagation.h",
    ],
    deps = [
        "//pytorch_blade/common_utils:torch_blade_macros",
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
