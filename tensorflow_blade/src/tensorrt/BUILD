load("@local_config_tf//:build_defs.bzl", "tf_copts")

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
    deps = [
        "//src/util:tf_allocator_util",
        "//src/util:logging",
        "@local_config_cuda//cuda:cuda_headers",
        "@blade_config_tensorrt//:nvinfer_static",
        "@blade_config_tensorrt//:nvinferplugin",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
    visibility = ["//visibility:private"]
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
        # "@local_config_cuda//cuda:nvrtc",
        # "@blade_config_tensorrt//:nvinfer_static",
        "@blade_config_tensorrt//:nvonnxparser",
        "@local_config_cuda//cuda:cudart_static",
        # TODO(lanbo.llb): Add a static version for cudnn/cublas ???
        "@local_config_cuda//cuda:cudnn",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cublasLt",
        "//src/util:logging",
    ],
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
)

exports_files([
    "pybind_functions.h",
    "pybind_functions.cpp",
])
