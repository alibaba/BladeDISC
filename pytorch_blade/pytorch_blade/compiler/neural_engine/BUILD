package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "torch_blade_neural_engine",
    visibility = [
        "//visibility:public",
    ],
    hdrs = [
        "neural_engine.h"
    ],
    srcs = [
        "neural_engine.cpp"
    ],
    alwayslink = True,
)

filegroup(
    name = "torch_blade_neural_engine_py_srcs",
    srcs = ["pybind_functions.cpp",],

)

filegroup(
    name = "torch_blade_neural_engine_py_hdrs",
    srcs = ["pybind_functions.h"],
)
