load("@local_config_tf//:build_defs.bzl", "tf_copts")
load("@local_config_cuda_supplement//:build_defs.bzl", "if_has_cublaslt", "if_has_cudnn_static")
load("@local_config_trt//:build_defs.bzl", "if_has_myelin")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensorrt_tf_impl",
    srcs = [
        "bridge/tensorrt_tf_allocator.cc",
        "bridge/tensorrt_tf_resource_mgr.cc",
    ],
    hdrs = [
        "bridge/tensorrt_tf_allocator.h",
        "bridge/tensorrt_tf_resource_mgr.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,  # this target only allow static linking, no .so file generated.
    visibility = ["//visibility:private"],
    deps = [
        "//src/util:logging",
        "//src/util:tf_allocator_util",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@local_config_trt//:nvinfer_static",
        "@local_config_trt//:nvinferplugin_static",
    ],
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
)

cc_library(
    name = "tensorrt_bridge_impl",
    srcs = [
        "bridge/tensorrt_common.cpp",
        "bridge/tensorrt_flags.cpp",
        "bridge/tensorrt_logger.cpp",
        "bridge/tensorrt_onnx_parser.cpp",
    ],
    hdrs = [
        "bridge/macros.h",
        "bridge/tensorrt_common.h",
        "bridge/tensorrt_flags.h",
        "bridge/tensorrt_logger.h",
        "bridge/tensorrt_onnx_parser.h",
    ],
    copts = [
        "-DWITHOUT_ONNX_UTIL",
        "-DUSE_TF_ALLOCATOR",
    ],
    linkstatic = 1,  # this target only allow static linking, no .so file generated.
    deps = [
        ":tensorrt_tf_impl",
        "@local_config_trt//:nvonnxparser_static",
        "@local_config_cuda//cuda:cudart_static",
        "@local_config_cuda_supplement//:cublas_static",
        "@local_config_cuda_supplement//:culibos_static",
        "//src/util:logging",
    ] + if_has_cudnn_static(
        [
            "@local_config_cuda_supplement//:cudnn_static",
        ],
        if_false = [
            # cuda 10.2 has no libcudnn_static.a
            "@local_config_cuda_supplement//:cudnn_adv_infer_static",
            "@local_config_cuda_supplement//:cudnn_cnn_infer_static",
            "@local_config_cuda_supplement//:cudnn_cnn_train_static",
            "@local_config_cuda_supplement//:cudnn_ops_infer_static",
            "@local_config_cuda_supplement//:cudnn_ops_train_static",
        ],
    ) + if_has_cublaslt([
        "@local_config_cuda_supplement//:cublasLt_static",
    ]) + if_has_myelin([
        "@local_config_cuda_supplement//:nvrtc",
        "@local_config_trt//:myelin_static",
    ]),
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
)

filegroup(
    name = "pybind_srcs",
    srcs = [
        "pybind_functions.h",
        "pybind_functions.cpp",
    ]
)

