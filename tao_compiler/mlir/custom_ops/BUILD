load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_shared_object",
    "tf_cc_test",
    "tf_copts",
    "tf_gpu_kernel_library",
    "tf_gpu_library",
    "tf_native_cc_binary",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_default_copts",
    "if_cuda",
    "if_cuda_is_configured",
    "cuda_library",
)
load("@local_config_rocm//rocm:build_defs.bzl","if_dcu", "if_rocm_is_configured")
load(
    "@com_google_protobuf//:protobuf.bzl",
    "cc_proto_library",
)
load("//mlir/util:util.bzl",
     "disc_cc_library",
     "if_cuda_or_rocm",
     "if_platform_alibaba",
     "if_blade_gemm",
     "if_mkldnn",
     "if_disc_aarch64",
     "if_disc_x86",
     "if_torch_disc",
     "if_internal_serving",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

tf_gpu_kernel_library(
    name = "dynamic_sort_kernel",
    srcs = [
        "custom_library/dynamic_sort.cu.cc",
        "custom_library/tf_topk.cu.h",
    ],
    hdrs = [
        "custom_library/dynamic_sort.h",
        "custom_library/gpu_helper.h"
    ],
    copts = if_rocm_is_configured([
        "-DTENSORFLOW_USE_ROCM=1",
    ]) + cuda_default_copts(),
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/types:optional",
    ] + if_cuda_is_configured([
        "@cub_archive//:cub",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

tf_gpu_library(
    name = "dynamic_sort",
    srcs = [
        "dynamic_sort_impl.cc",
    ],
    hdrs = [
        "dynamic_sort_impl.h",
    ],
    deps = [
        "//mlir/ral:ral_base_context_lib",
        ":dynamic_sort_kernel",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_driver",
    ]) + if_rocm_is_configured([
        "//tensorflow/stream_executor/rocm:rocm_driver",
    ]),
    alwayslink = 1,
)

tf_cc_test(
    name = "philox_random_test",
    size = "small",
    srcs = [
        "context/custom_library/philox_random_test.cc",
        "context/custom_library/philox_random.h"
    ],
    deps = [
        "//tensorflow/core:test_main",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
    ],
)

tf_gpu_kernel_library(
    name = "random_gpu_lib",
    srcs = [
        "custom_library/random_gpu.cu.cc",
    ],
    hdrs = [
        "custom_library/random.h",
        "custom_library/philox_random.h",
    ],
    copts = if_rocm_is_configured([
        "-DTENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "random",
    srcs = [
        "random_impl.cc",
    ],
    hdrs = [
    ],
    deps = [
        ":random_gpu_lib",
        "//mlir/ral:ral_base_context_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:stream_executor_headers_lib",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured([
        "@org_tensorflow//tensorflow/compiler/xla/stream_executor/cuda:all_runtime",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@org_tensorflow//tensorflow/stream_executor:rocm_platform",
        "@org_tensorflow//tensorflow/stream_executor/rocm:rocm_driver",
        "@local_config_rocm//rocm:rocm_headers",
    ]),
    alwayslink = 1,
)

tf_gpu_kernel_library(
    name = "transpose_gpu_lib",
    srcs = [
        "custom_library/transpose_gpu.cu.cc",
        "custom_library/tf_transpose.cu.h",
    ],
    hdrs = [
        "custom_library/transpose.h",
        "custom_library/dimensions.h",
        "custom_library/gpu_helper.h"
    ],
    includes = [
        "",
    ],
    copts = if_cuda_is_configured([
        "-DGOOGLE_CUDA=1"
    ]) + if_cuda_or_rocm([
        "-DTAO_RAL_USE_STREAM_EXECUTOR"
    ]),
    deps = [
        "//mlir/ral:ral_base_context_lib",
    ] + if_cuda_is_configured([
        "@cub_archive//:cub",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

cc_library(
    name = "transpose",
    srcs = [
        "transpose_impl.cc",
    ],
    deps = [
        ":transpose_gpu_lib",
        "//mlir/ral:ral_base_context_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:stream_executor_headers_lib",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured([
        "@org_tensorflow//tensorflow/compiler/xla/stream_executor/cuda:all_runtime",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@org_tensorflow//tensorflow/stream_executor:rocm_platform",
        "@org_tensorflow//tensorflow/stream_executor/rocm:rocm_driver",
        "@local_config_rocm//rocm:rocm_headers",
    ]),
    copts = if_cuda_is_configured([
        "-DGOOGLE_CUDA=1"
    ]) + if_dcu([
        "-DTENSORFLOW_USE_DCU=1"
    ]) + if_cuda_or_rocm(["-DTAO_RAL_USE_STREAM_EXECUTOR"]),
    alwayslink = 1,
)

tf_cc_shared_object(
    name = "libdisc_custom_ops.so",
    linkopts = select({
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "$(location //mlir/ral:context_version_scripts.lds)",
        ],
    }),
    deps = [
        "//mlir/ral:context_version_scripts.lds",
    ] + if_cuda_or_rocm([
        ":dynamic_sort",
        ":transpose",
        ":random",
    ])
)
cc_library(
    name = "disc_custom_ops_lib",
    data = [
        ':libdisc_custom_ops.so',
    ],
    srcs = ["libdisc_custom_ops.so"],
    includes = [
        ".",
    ],
    visibility = ["//visibility:public"],
)

########### Tao Bridge for Tensorflow ##########

cuda_library(
    name = "dynamic_sort_kernel_bridge",
    srcs = [
        "custom_library/dynamic_sort.cu.cc",
        "custom_library/tf_topk.cu.h",
    ],
    hdrs = [
        "custom_library/dynamic_sort.h",
        "custom_library/gpu_helper.h"
    ],
    copts = [
        "-DDISC_BUILD_FROM_TF_BRIDGE"
    ] + if_rocm_is_configured([
        "-DTENSORFLOW_USE_ROCM=1",
        "-x rocm",
    ]) + cuda_default_copts(),
    deps = [
        "//mlir/ral:tf_deps",
    ] + if_cuda_is_configured([
        "@cub_archive//:cub",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

cc_library(
    name = "dynamic_sort_bridge",
    srcs = [
        "dynamic_sort_impl.cc",
    ],
    hdrs = [
        "dynamic_sort_impl.h",
    ],
    copts = [
        "-DDISC_BUILD_FROM_TF_BRIDGE"
    ] + if_cuda_is_configured([
        "-DGOOGLE_CUDA=1",
    ]),
    deps = [
        "//mlir/ral:ral_bridge",
        ":dynamic_sort_kernel_bridge",
        "//mlir/ral:tf_deps",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_driver",
    ]),
    alwayslink = 1,
)
cuda_library(
    name = "random_gpu_lib_bridge",
    srcs = [
        "custom_library/random_gpu.cu.cc",
    ],
    hdrs = [
        "custom_library/random.h",
        "custom_library/philox_random.h",
    ],
    copts = if_rocm_is_configured([
        "-DTENSORFLOW_USE_ROCM=1",
        "-x rocm",
    ]) + cuda_default_copts(),
    deps = if_cuda_is_configured([
        "@cub_archive//:cub",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

cuda_library(
    name = "transpose_gpu_lib_bridge",
    srcs = [
        "custom_library/transpose_gpu.cu.cc",
        "custom_library/tf_transpose.cu.h",
    ],
    hdrs = [
        "custom_library/transpose.h",
        "custom_library/dimensions.h",
        "custom_library/gpu_helper.h"
    ] + if_cuda_is_configured([
        "//mlir/ral:context/stream_executor_based_impl.h"
    ]),
    copts = [
        "-DDISC_BUILD_FROM_TF_BRIDGE"
    ] + if_rocm_is_configured([
        "-DTENSORFLOW_USE_ROCM=1",
        "-x rocm",
    ]) + if_cuda_or_rocm([
        "-DTAO_RAL_USE_STREAM_EXECUTOR"
    ]) + cuda_default_copts(),
    deps = [
        "//mlir/ral:ral_bridge",
    ] + if_cuda_is_configured([
        "@cub_archive//:cub",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]) + if_blade_gemm([
        "@blade_gemm//:blade_gemm",
    ]),
)


cc_library(
    name = "transpose_bridge",
    srcs = [
        "transpose_impl.cc",
    ],
    copts = [
        "-DDISC_BUILD_FROM_TF_BRIDGE",
    ] + if_rocm_is_configured([
        "-DTENSORFLOW_USE_ROCM=1",
        "-x rocm",
    ]) + if_dcu([
        "-DTENSORFLOW_USE_DCU=1"
    ]) + if_cuda_or_rocm([
        "-DTAO_RAL_USE_STREAM_EXECUTOR"
    ]) + cuda_default_copts(),
    deps = [
        "//mlir/ral:ral_context",
        "//mlir/ral:ral_gpu_driver",
        "//mlir/ral:ral_cpu_driver",
        "//mlir/ral:ral_logging",
        "//mlir/ral:context_util",
        "//mlir/ral:tf_deps",
        ":transpose_gpu_lib_bridge",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "random_bridge",
    srcs = [
        "random_impl.cc",
    ],
    deps = [
        "//mlir/ral:ral_bridge",
        "//mlir/ral:tf_deps",
        ":random_gpu_lib_bridge",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "disc_custom_ops_bridge",
    deps = [
        "//mlir/ral:common_context_bridge",
        "//mlir/ral:ral_tf_context_bridge",
    ] + if_cuda_or_rocm([
        ":dynamic_sort_bridge",
        ":random_bridge",
        ":transpose_bridge",
    ]),
    alwayslink = 1,
)