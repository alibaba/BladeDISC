load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

filegroup(
    name = "torch_blade_common_srcs",
    srcs = glob(["**/*.cpp"]),
)

filegroup(
    name = "torch_blade_common_hdrs",
    srcs = glob(["**/*.h"]),
)

cc_library(
    name = "torch_blade_common",
    srcs = [":torch_blade_common_srcs"],
    hdrs = [
        ":torch_blade_common_hdrs",
    ],
    includes = ["../"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@local_org_torch//:libtorch",
    ],
    alwayslink = True,
)
