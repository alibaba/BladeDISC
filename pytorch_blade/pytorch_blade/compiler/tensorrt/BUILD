load("//bazel:build_defs.bzl", "if_tensorrt_static_linked")
load("@local_config_cuda_supplement//:build_defs.bzl", "if_has_cublaslt", "if_has_cudnn_static")
load("@tensorrt//:build_defs.bzl", "if_has_myelin")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensorrt_bridge_impl",
    srcs = [
        "bridge/tensorrt_common.cpp",
        "bridge/tensorrt_flags.cpp",
        "bridge/tensorrt_logger.cpp",
        "bridge/tensorrt_onnx_parser.cpp",
        "bridge/tensorrt_calibrator.cpp",
    ],
    hdrs = [
        "bridge/tensorrt_common.h",
        "bridge/tensorrt_flags.h",
        "bridge/tensorrt_logger.h",
        "bridge/tensorrt_onnx_parser.h",
        "bridge/tensorrt_calibrator.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "//pytorch_blade/common_utils:torch_blade_common",
        "//pytorch_blade/compiler/backends:torch_blade_backends",
    ] + if_tensorrt_static_linked(
        [
            "@tensorrt//:nvinfer_static",
            "@tensorrt//:nvinferplugin_static",
            "@tensorrt//:nvonnxparser_static",
            "@local_config_cuda_supplement//:cublas_static_whole_archived",
            "@local_config_cuda_supplement//:culibos_static_whole_archived",
        ] + if_has_cudnn_static(
            [
                "@local_config_cuda_supplement//:cudnn_static",
            ],
            if_false = [
                # cuda 10.2 has no libcudnn_static.a
                "@local_config_cuda_supplement//:cudnn_adv_infer_static",
                "@local_config_cuda_supplement//:cudnn_cnn_infer_static_whole_archived",
                "@local_config_cuda_supplement//:cudnn_cnn_train_static",
                "@local_config_cuda_supplement//:cudnn_ops_infer_static",
                "@local_config_cuda_supplement//:cudnn_ops_train_static",
            ],
        ) + if_has_cublaslt([
            "@local_config_cuda_supplement//:cublasLt_static_whole_archived",
        ]) + if_has_myelin([
            "@local_config_cuda_supplement//:nvrtc",
            "@tensorrt//:myelin_static",
        ]),
        if_false = [
            "@tensorrt//:nvinfer",
            "@tensorrt//:nvinferplugin",
            "@tensorrt//:nvonnxparser",
        ],
    ),
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
)

cc_library(
    name = "tensorrt_runtime_impl",
    srcs = [
        "tensorrt_engine.cpp",
        "tensorrt_engine_context.cpp",
    ],
    hdrs = [
        "tensorrt_engine.h",
        "tensorrt_engine_context.h",
    ],
    deps = [
        ":tensorrt_bridge_impl",
    ],
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
)

exports_files([
    "pybind_functions.h",
    "pybind_functions.cpp",
])
