load("//bazel:build_defs.bzl", "if_tensorrt_static_linked")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensorrt_bridge_impl",
    srcs = [
        "bridge/tensorrt_common.cpp",
        "bridge/tensorrt_flags.cpp",
        "bridge/tensorrt_logger.cpp",
        "bridge/tensorrt_onnx_parser.cpp",
    ],
    hdrs = [
        "bridge/tensorrt_common.h",
        "bridge/tensorrt_flags.h",
        "bridge/tensorrt_logger.h",
        "bridge/tensorrt_onnx_parser.h",
    ],
    deps = [
        "//src/common_utils:torch_blade_common",
        "//src/compiler/backends:torch_blade_backends",
    ] + if_tensorrt_static_linked(
        [
            "@tensorrt//:nvinfer_static",
            "@tensorrt//:nvinferplugin_static",
            "@tensorrt//:nvonnxparser_static",
        ],
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
