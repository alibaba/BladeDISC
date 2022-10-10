load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

filegroup(
    name = "torch_blade_mhlo_converter_impl_srcs",
    srcs = glob(["**/*.cpp"]),
)

cc_library(
    name = "torch_blade_mhlo_converter_impl",
    srcs = [":torch_blade_mhlo_converter_impl_srcs"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//pytorch_blade/common_utils:torch_blade_common",
        "//pytorch_blade/compiler/mlir/converters:torch_blade_mhlo_converter",
        "@local_org_torch//:libtorch",
        "@org_mhlo_builder//csrc:mhlo_library",
    ],
    copts = select({
       "//:enable_cuda": ["-DTORCH_BLADE_BUILD_WITH_CUDA"],
       "//conditions:default": []}),
    alwayslink = True,
)
