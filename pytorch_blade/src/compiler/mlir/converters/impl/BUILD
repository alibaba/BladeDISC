
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

filegroup(
    name = "torch_blade_mhlo_converter_impl_srcs",
    srcs = glob(["**/*.cpp"]),
)

cc_library(
    name = "torch_blade_mhlo_converter_impl",
    srcs = [":torch_blade_mhlo_converter_impl_srcs"],
    deps = [
        "//src/common_utils:torch_blade_common",
        "//src/compiler/mlir/converters:torch_blade_mhlo_converter",
        "@org_mhlo_builder//csrc:mhlo_library",
        "@local_org_torch//:libtorch",
    ],
    alwayslink = True,
    visibility = [
        "//visibility:public",
    ],
)

