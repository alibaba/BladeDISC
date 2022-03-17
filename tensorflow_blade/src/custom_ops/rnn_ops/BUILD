load("//src/custom_ops:build_defs.bzl", "tf_blade_cuda_library")

package(default_visibility = ["//visibility:public"])

tf_blade_cuda_library(
    name = "rnn_ops",
    srcs = [
        "fused_lstm_element_wise_op.cc",
        "fused_lstm_element_wise_op_gpu.cu.cc",
        "fused_lstm_element_wise_op_kernels.cu.h",
        "rnn_ops.cc",
    ],
)
