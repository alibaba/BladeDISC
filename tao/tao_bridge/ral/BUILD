package(default_visibility = ["//visibility:public"])
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@local_config_tf//:build_defs.bzl", "tf_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "cuda_library")

cc_proto_library(
    name = "compile_metadata",
    srcs = [
        "tensorflow/compiler/mlir/xla/compile_metadata.proto",
    ],
)

filegroup(
    name = "ral_headers",
    srcs = glob(["tensorflow/**/*.h"]),
)

cc_library(
    name = "bridge_ral",
    srcs = [
        "tensorflow/compiler/mlir/xla/compile_metadata.pb.cc",
        "tensorflow/compiler/mlir/xla/ral/context/common_context_impl.cc",
        "tensorflow/compiler/mlir/xla/ral/context/stream_executor_based_impl.cc",
        "tensorflow/compiler/mlir/xla/ral/context/tensorflow/tf_context_impl.cc",
        "tensorflow/compiler/mlir/xla/ral/context/tensorflow/tf_kernel_impl.cc",
        "tensorflow/compiler/mlir/xla/ral/device/cpu/cpu_driver.cc",
        "tensorflow/compiler/mlir/xla/ral/device/gpu/gpu_driver.cc",
        "tensorflow/compiler/mlir/xla/ral/ral_api.cc",
        "tensorflow/compiler/mlir/xla/ral/ral_context.cc",
        "tensorflow/compiler/mlir/xla/ral/ral_helper.cc",
        "tensorflow/compiler/mlir/xla/ral/ral_logging.cc",
    ],
    hdrs = [
        ":ral_headers",
    ],
    copts = [
        "-DDISC_BUILD_FROM_TF_BRIDGE",
        "-Iexternal/org_tao/tao_bridge/ral",
    ] + if_cuda([
        "-DTAO_RAL_USE_STREAM_EXECUTOR",
    ]) + tf_copts(),
    deps = [
        ":compile_metadata",
        # ":bridge_ral_gpu",
        "@local_config_tf//:tf_header_lib",
        "@local_config_tf//:libtensorflow_framework",
    ],
    linkstatic = 1,
    # strip_include_prefix = "external/org_tao/tao_bridge/ral",
    # include_prefix = "tao_bridge/ral",
    # includes = ["tao_bridge/ral"]
)

cuda_library(
    name = "bridge_ral_cuda",
    srcs = [
        "tensorflow/compiler/mlir/xla/ral/context/dynamic_sort_impl.cc",
        "tensorflow/compiler/mlir/xla/ral/context/random_impl.cc",
        "tensorflow/compiler/mlir/xla/ral/context/common_context_impl_cuda.cc",
        "tensorflow/compiler/mlir/xla/ral/context/custom_library/dynamic_sort.cu.cc",
        "tensorflow/compiler/mlir/xla/ral/context/custom_library/random_gpu.cu.cc",
    ],
    copts = [
        "-DGOOGLE_CUDA=1",
        "-DDISC_BUILD_FROM_TF_BRIDGE",
        "-Iexternal/org_tao/tao_bridge/ral",
        "-Wno-deprecated-gpu-targets",
    ] + tf_copts(),
    linkopts = [
        "-lstdc++",
    ],
    deps = [
        ":bridge_ral",
        ":compile_metadata",
        "@cub_archive//:cub",
        "@local_config_tf//:tf_header_lib",
        "@local_config_tf//:libtensorflow_framework",
    ],
    linkstatic = 1,
)
