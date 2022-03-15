load("//src/custom_ops:build_defs.bzl", "tf_blade_cuda_library")

package(default_visibility = ["//visibility:public"])

tf_blade_cuda_library(
    name = "non_max_suppression",
    srcs = [
        "cuda_common.cu.h",
        "gpu_non_max_suppression.cu.h",
        "gpu_non_max_suppression_op.cu.cc",
    ],
)
