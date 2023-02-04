load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    srcs = [
        "utils.cpp"
    ],
    hdrs = [
        "utils.h"
    ],
    deps = [
        "@mlir-hlo//:mlir_hlo",
        "@llvm-project//mlir:IR",
        "@llvm-project//llvm:Support",
        "@org_disc_compiler//mlir/disc:disc_pdl_utils",
        "@torch-mlir//:TorchMLIRTorchDialect",
    ]
)

cc_binary(
    name = "torch-disc-pdll",
    srcs = [
        "torch-disc-pdll.cpp"
    ],
    deps = [
        ":utils",
        "@torch-mlir//:TorchMLIRInitAll",
        "@llvm-project//mlir:AllPassesAndDialects",
    ]
)
