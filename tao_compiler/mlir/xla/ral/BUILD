# RAL implementations

load(
    "//tensorflow:tensorflow.bzl",
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
)
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load(
    "@com_google_protobuf//:protobuf.bzl",
    "cc_proto_library",
)
load("//tensorflow/compiler/mlir/disc:disc.bzl",
     "disc_cc_library",
     "if_cuda_or_rocm",
     "if_platform_alibaba",
     "if_patine",
     "if_mkldnn",
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
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/compiler/xla/...",
        "//third_party/iree/...",
        "//third_party/mlir_edge/...",
        "//third_party/tf_runtime/tools/tf_kernel_gen/...",
    ],
)

cc_proto_library(
    name = "compile_metadata",
    srcs = ["compile_metadata.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "ral_logging",
    srcs = [
        "ral_logging.cc",
    ],
    hdrs = [
        "ral_logging.h",
    ],
    deps = [
    ],
    alwayslink = 1,
)

cc_library(
    name = "ral_context",
    srcs = [
        "ral_context.cc",
        "ral_helper.cc",
    ],
    hdrs = [
        "ral_context.h",
        "ral_helper.h",
        "ral_driver.h",
        "ral_base.h",
    ],
    deps = [
        ":ral_logging",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ral_cpu_driver",
    srcs = ["device/cpu/cpu_driver.cc"],
    hdrs = ["device/cpu/cpu_driver.h"],
    deps = [
        ":ral_context",
        ":ral_logging"
    ],
    alwayslink = 1,
)

cc_library(
    name = "ral_gpu_driver",
    srcs = ["device/gpu/gpu_driver.cc"],
    hdrs = ["device/gpu/gpu_driver.h"],
    deps = [
        ":ral_context",
        ":ral_logging"
    ],
    alwayslink = 1,
)

cc_library(
    name = "ral_library",
    srcs = ["ral_api.cc"],
    hdrs = ["ral_api.h"],
    deps = [
        ":ral_context",
        ":ral_logging",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ral_tf_context",
    srcs = [
        "context/tensorflow/tf_context_impl.cc",
        "context/tensorflow/tf_kernel_impl.cc",
    ],
    hdrs = ["context/tensorflow/tf_context_impl.h"],
    deps = [
        ":context_util",
        ":common_context",
        ":ral_context",
        ":ral_cpu_driver",
        ":ral_gpu_driver",
        ":ral_library",
        "//tensorflow/core:framework",
        "//tensorflow/stream_executor",
        "//tensorflow/core:lib",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "context_util",
    srcs = [],
    hdrs = ["context/context_util.h"],
    deps = [
        ":ral_context",
        ":ral_logging"
    ],
    alwayslink = 1,
)

cc_library(
    name = "real_disc_patine_client_deps",
    srcs = [],
    hdrs = [],
    copts = ["-Itao/third_party/PatineClient/build/install/include"],
    linkopts = [
      "-Ltao/third_party/PatineClient/build/install/lib64",
      "-Wl,--start-group tao/third_party/PatineClient/build/intel/lib/libmkl_intel_ilp64.a tao/third_party/PatineClient/build/intel/lib/libmkl_gnu_thread.a tao/third_party/PatineClient/build/intel/lib/libmkl_core.a -Wl,--end-group",
    ],
    deps = ["//tao/third_party/PatineClient:patine_client"],
)


cc_library(
    name = "disc_patine_client_deps",
    deps = if_platform_alibaba([":real_disc_patine_client_deps"])
)

cc_library(
    name = "disc_mkl",
    hdrs = glob([
        "context/mkldnn/ideep/*.hpp",
        "context/mkldnn/ideep/ideep/*.hpp",
        "context/mkldnn/ideep/ideep/operators/*.hpp",
    ]),
    linkopts = [
      "-L$(location //tao/third_party/mkldnn:build)/install/lib64",
      "-L$(location //tao/third_party/mkldnn:build)/intel/lib",
      "-Wl,--start-group",
      "$(location //tao/third_party/mkldnn:build)/install/lib64/libdnnl.a",
      "-Wl,--end-group",
      "-Wl,--start-group",
      "$(location //tao/third_party/mkldnn:build)/intel/lib/libmkl_intel_ilp64.a",
      "$(location //tao/third_party/mkldnn:build)/intel/lib/libmkl_gnu_thread.a",
      "$(location //tao/third_party/mkldnn:build)/intel/lib/libmkl_core.a",
      "-Wl,--end-group",
    ],
    data = ["//tao/third_party/mkldnn:build"],
    deps = [
        "//tao/third_party/mkldnn:onednn",
    ],
)


cc_library(
    name = "common_context",
    srcs = [
        "context/common_context_impl.cc",
    ] + if_cuda_or_rocm([
        "context/common_context_impl_cuda.cc",
        "context/stream_executor_based_impl.cc",
    ]) + if_patine([
        "context/common_context_impl_patine.cc",
    ]) + if_mkldnn([
        "context/common_context_impl_mkldnn.cc",
    ]),
    hdrs = [
        "context/common_context_impl.h",
    ] + if_cuda_or_rocm([
        "context/stream_executor_based_impl.h",
    ]),
    copts = [
        # "-DTF_1_12",
        "-fopenmp",
    ] + if_cuda_or_rocm(["-DTAO_RAL_USE_STREAM_EXECUTOR"]),
    linkopts = [
        "-fopenmp",
        "-ldl"
    ],
    deps = [
        ":ral_context",
        ":ral_cpu_driver",
        ":ral_logging",
        ":context_util",
        ":compile_metadata",
        "@com_google_absl//absl/strings"
    ] + if_cuda_is_configured([
        "//tensorflow/stream_executor:cuda_platform",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "//tensorflow/stream_executor:rocm_platform",
        "//tensorflow/stream_executor/rocm:rocm_driver",
        "@local_config_rocm//rocm:rocm_headers",
    ]) + if_cuda_or_rocm([
        ":ral_gpu_driver",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_headers_lib",
        "//tensorflow/core:framework",
    ]) + if_patine([
        ":disc_patine_client_deps",
    ]) + if_mkldnn([
        ":disc_mkl",
    ]),
    alwayslink = 1,
)

tf_gpu_library(
    name = "dynamic_sort",
    srcs = [
        "context/dynamic_sort_impl.cc",
    ],
    hdrs = [
        "context/dynamic_sort_impl.h",
    ],
    deps = [
        ":ral_context",
        ":ral_gpu_driver",
        ":ral_logging",
        ":context_util",
        ":common_context",
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
        "context/custom_library/random_gpu.cu.cc",
    ],
    hdrs = [
        "context/custom_library/random.h",
        "context/custom_library/philox_random.h",
    ],
    copts = if_rocm_is_configured([
        "-DTENSORFLOW_USE_ROCM",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "random",
    srcs = [
        "context/random_impl.cc",
    ],
    hdrs = [
    ],
    #copts = ["-DTF_1_12"],
    deps = [
        ":ral_context",
        ":ral_gpu_driver",
        ":ral_cpu_driver",
        ":ral_logging",
        ":context_util",
        ":random_gpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_headers_lib",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured([
        "//tensorflow/stream_executor:cuda_platform",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "//tensorflow/stream_executor:rocm_platform",
        "//tensorflow/stream_executor/rocm:rocm_driver",
        "@local_config_rocm//rocm:rocm_headers",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "init_stream_executor",
    srcs = [
        "context/init_stream_executor.cc",
        "context/base/cuda/cuda_stream.cc"
    ],
    hdrs = [
        "context/init_stream_executor.h",
        "context/base/cuda/cuda_stream.h"
    ],
    deps = [
        ":common_context",
        ":ral_logging",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured([
        "//tensorflow/stream_executor:cuda_platform",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "//tensorflow/stream_executor:rocm_platform",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "ral_base_cpu_context_impl",
    srcs = [
        "context/base/cpu/cpu_context_impl.cc",
        "context/base/base_context.cc",
    ],
    hdrs = [
        "context/base/cpu/cpu_context_impl.h",
        "context/base/base_context.h",
    ] + if_cuda_or_rocm([
        "context/stream_executor_based_impl.h"
    ]),
    copts = if_cuda_or_rocm([
        "-DTAO_RAL_USE_STREAM_EXECUTOR",
    ]),
    deps = [
        ":context_util",
        ":common_context",
        ":ral_context",
        ":ral_cpu_driver",
        ":ral_library",
        ":ral_logging",
        "@com_google_absl//absl/strings",
        "@curl",
    ] + if_cuda_or_rocm([
        "//tensorflow/core:stream_executor_headers_lib",
    ]),
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

disc_cc_library(
    name = "ral_base_cuda_context_impl",
    srcs = [
        "context/base/cuda/cuda_context_impl.cc",
    ],
    hdrs = [
        "context/base/cuda/cuda_context_impl.h",
    ],
    copts = ["-DTAO_RAL_USE_STREAM_EXECUTOR"],
    deps = [
        ":ral_base_cpu_context_impl",
        ":context_util",
        ":common_context",
        ":dynamic_sort",
        ":init_stream_executor",
        ":ral_context",
        ":ral_cpu_driver",
        ":ral_gpu_driver",
        ":ral_library",
        ":ral_logging",
        ":random",
        ":compile_metadata",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@curl",
    ] + if_rocm_is_configured([
        "//tensorflow/stream_executor/rocm:rocm_platform",
        "//tensorflow/stream_executor/rocm:rocm_driver",
    ]) + if_cuda_is_configured([
        "//tensorflow/stream_executor/cuda:cuda_platform",
        "@local_config_cuda//cuda:cuda_driver",
    ]),
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

tf_cc_shared_object(
    name = "libral_base_context.so",
    linkopts = select({
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "$(location //tensorflow/compiler/mlir/xla/ral:context_version_scripts.lds)",
        ],
    }),
    deps = [
        ":ral_base_cpu_context_impl",
        "//tensorflow/compiler/mlir/xla/ral:context_version_scripts.lds",
    ] + if_cuda_or_rocm([
        ":ral_base_cuda_context_impl",
    ]))

cc_library(
    name = "ral_base_context_lib",
    data = [
        ':libral_base_context.so',
    ],
    srcs = ["libral_base_context.so"],
    includes = [
        ".",
    ],
    visibility = ["//visibility:public"],
)

tf_gpu_kernel_library(
    name = "dynamic_sort_kernel",
    srcs = [
        "context/custom_library/dynamic_sort.cu.cc",
        "context/custom_library/tf_topk.cu.h",
    ],
    hdrs = [
        "context/custom_library/dynamic_sort.h",
    ],
    copts = if_rocm_is_configured([
        "-DTENSORFLOW_USE_ROCM",
    ]),
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
