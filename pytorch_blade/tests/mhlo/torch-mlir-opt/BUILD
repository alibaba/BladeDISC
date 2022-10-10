load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "torch-mlir-opt",
    srcs = [
        "torch-mlir-opt.cpp"
    ],
    deps = [
        "//pytorch_blade/torch-mlir:DiscTorchMLIRTorchToMhlo",
        "@torch-mlir//:TorchMLIRInitAll",
        "@torch-mlir//:TorchMLIRTorchDialect",
        "@torch-mlir//:TorchMLIRTorchPasses",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib"
    ]
)
