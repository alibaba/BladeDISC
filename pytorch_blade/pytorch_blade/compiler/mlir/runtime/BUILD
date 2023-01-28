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
        "//pytorch_blade/common_utils:torch_blade_common",
        "//pytorch_blade/compiler/jit:torch_blade_jit",
        "@org_disc_compiler//mlir/xla/ral:ral_base_context_lib",
        "@local_org_torch//:ATen",
        "@local_org_torch//:libtorch",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
    copts = select({
       "//:enable_rocm": ["-DTORCH_BLADE_USE_ROCM -DTORCH_BLADE_BUILD_WITH_CUDA "],
       "//:enable_cuda": ["-DTORCH_BLADE_BUILD_WITH_CUDA"],
       "//conditions:default": []}),
    alwayslink = True,
    visibility = [
        "//visibility:public",
    ]
)

cc_library(
    name = "torch_blade_disc_rt",
    srcs = ["disc_engine.cpp"],
    hdrs = [
        "disc_engine.h",
        "ral_context.h",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":torch_blade_ral_context",
	    "//pytorch_blade/compiler/backends:torch_blade_backends",
        "@local_org_torch//:libtorch",
    ],
    copts = select({
       "//:enable_rocm": ["-DTORCH_BLADE_USE_ROCM -DTORCH_BLADE_BUILD_WITH_CUDA "],
       "//:enable_cuda": ["-DTORCH_BLADE_BUILD_WITH_CUDA"],
       "//conditions:default": []}),
    alwayslink = True,
)
