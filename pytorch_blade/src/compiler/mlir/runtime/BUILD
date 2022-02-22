load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda_is_configured",
)

cc_library(
    name = "torch_blade_ral_context",
    srcs = ["ral_context.cpp"],
    hdrs = ["ral_context.h"],
    deps = [
        "//src/common_utils:torch_blade_common",
        "//src/compiler/jit:torch_blade_jit",
        "@org_tensorflow//tensorflow/compiler/mlir/xla/ral:ral_base_context_lib",
        "@local_org_torch//:ATen",
        "@local_org_torch//:libtorch",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
    alwayslink = True,
)

cc_library(
    name = "torch_blade_disc_rt",
    srcs = ["disc_engine_class.cpp"],
    hdrs = [
        "disc_engine_class.h",
        "ral_context.h",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":torch_blade_ral_context",
        "@local_org_torch//:libtorch",
    ],
    alwayslink = True,
)
