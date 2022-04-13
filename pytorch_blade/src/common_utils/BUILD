load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "torch_blade_logging",
    hdrs = ["logging.h"],
    deps = [
        "@local_org_torch//:libtorch",
        "@org_tensorflow//tensorflow/core/platform:logging",
    ],
)

cc_library(
    name = "torch_blade_tmpfs",
    srcs = ["tempfs.cpp"],
    hdrs = ["tempfs.h"],
    deps = [
        ":torch_blade_logging",
        ":torch_blade_macros",
        "//src:torch_blade_include",
    ],
)

cc_library(
    name = "torch_blade_macros",
    hdrs = ["macros.h"],
)

cc_library(
    name = "torch_blade_utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h"],
    deps = [
        ":torch_blade_logging",
        ":torch_blade_macros",
        "//src:torch_blade_include",
        "@local_org_torch//:libtorch",
    ],
)

cc_library(
    name = "torch_blade_common",
    includes = ["../"],
    deps = [
        ":torch_blade_logging",
        ":torch_blade_macros",
        ":torch_blade_tmpfs",
        ":torch_blade_utils",
    ],
    alwayslink = True,
)

cc_test(
    name = "torch_blade_tmpfs_test",
    srcs = ["tempfs_test.cpp"],
    linkopts = [
        "-ldl",
    ],
    linkstatic = True,
    deps = [
        ":torch_blade_tmpfs",
        "@googltest//:gtest_main",
    ],
)
