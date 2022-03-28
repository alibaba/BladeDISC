package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "torch_blade_logging",
    hdrs = ["logging.h"],
    deps = [
       "@local_org_torch//:libtorch",
       "@org_tensorflow//tensorflow/core/platform:logging",]
)

cc_library(
    name = "torch_blade_tmpfs",
    srcs = ["tempfs.cpp"],
    hdrs = ["tempfs.h"],
    deps = [
       "//src:torch_blade_include",
       ":torch_blade_logging"]
)

cc_library(
    name = "torch_blade_macros",
    hdrs = ["macros.h"],
)

cc_library(
    name = "torch_blade_version",
    hdrs = ["version.h"],
)

cc_library(
    name = "torch_blade_utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h"],
    deps = [
      "//src:torch_blade_include",
      ":torch_blade_logging",
      ":torch_blade_macros",
      "@local_org_torch//:libtorch",
    ]
)

cc_library(
    name = "torch_blade_common",
    includes = ["../"],
    deps = [
        ":torch_blade_logging",
        ":torch_blade_tmpfs",
        ":torch_blade_macros",
        ":torch_blade_version",
        ":torch_blade_utils",
    ],
    alwayslink = True,
)
