package(default_visibility = ["//visibility:public"])

cc_library(
    name = "quantization_op",
    srcs = [
        "custom_op.cpp",
    ],
    hdrs = [
        "alias.h",
    ],
    deps = [
        "@local_org_torch//:libtorch",
        "//pytorch_blade/common_utils:torch_blade_common"
    ],
    alwayslink = True,
)

filegroup(
    name = "torch_blade_quantization_srcs",
    srcs = [
        "custom_op.cpp",
    ]
)

filegroup(
    name = "torch_blade_quantization_hdrs",
    srcs = [
        "alias.h",
    ]
)

filegroup(
    name = "torch_blade_quantization_py_srcs",
    srcs = [
        ":torch_blade_quantization_hdrs",
        "pybind_functions.cpp",
    ]
)

filegroup(
    name = "torch_blade_quantization_py_hdrs",
    srcs = [
        "pybind_functions.h",
    ]
)
