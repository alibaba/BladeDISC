package(default_visibility = ["//visibility:public"])

cc_library(
    name = "quantization_op",
    srcs = ["placeholder_op.cpp"],
    hdrs = ["placeholder_op.h"],
    deps = [
        "@local_org_torch//:libtorch",
    ],
    alwayslink = 1,
)

filegroup(
    name = "torch_blade_quantization_srcs",
    srcs = [
        "placeholder_op.cpp",
    ]
)

filegroup(
    name = "torch_blade_quantization_hdrs",
    srcs = [
        "placeholder_op.h",
    ]
)

filegroup(
    name = "torch_blade_quantization_py_srcs",
    srcs = [
        "pybind_functions.cpp",
    ]
)

filegroup(
    name = "torch_blade_quantization_py_hdrs",
    srcs = [
        "pybind_functions.h",
    ]
)

#exports_files([
#    "pybind_functions.h",
#    "pybind_functions.cpp",
#])
