load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_native_cc_binary",  "tf_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured",)
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured",)
load("@org_tensorflow//tensorflow/core/platform:build_config.bzl", "tf_proto_library",)
load("@org_tensorflow//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//mlir/util:util.bzl", "disc_cc_library")

# buildifier: disable=same-origin-load
load("@org_tensorflow//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

load(
    "@com_google_protobuf//:protobuf.bzl",
    "cc_proto_library",
)


package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    packages = [
        "//babelfish/device/...",
        "//learning/brain/experimental/mlir/...",
        "//learning/brain/experimental/swift_mlir/...",
        "//learning/brain/google/xla/kernels/...",
        "//learning/brain/swift/swift_mlir/...",
        "//mlir/...",
        "//third_party/iree/...",
        "//third_party/mlir_edge/...",
        "//third_party/tf_runtime/tools/tf_kernel_gen/...",
    ],
)

disc_cc_library(
    name = "mlir_test",
    srcs = ["mlir_test.cc"],
    hdrs = ["mlir_test.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//mlir/xla/ral:ral_base_context_lib",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cudart",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_driver",
    ]),
    alwayslink = 1,
)

disc_cc_library(
    name = "mlir_feature_test",
    testonly = 1,
    srcs = ["mlir_feature_test.cc"],
    hdrs = ["mlir_feature_test.h"],
    deps = [
        ":mlir_test",
        "@org_tensorflow//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)
