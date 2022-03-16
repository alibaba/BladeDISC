package(default_visibility = ["//visibility:public"])
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@local_config_tf//:build_defs.bzl", "tf_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")

cc_proto_library(
    name = "tao_compiler_input",
    srcs = [
        "tao_bridge/tao_compiler_input.proto",
    ],
)

cc_proto_library(
    name = "tao_compilation_result",
    srcs = [
        "tao_bridge/tao_compilation_result.proto",
    ],
)

cc_proto_library(
    name = "xla_activity",
    srcs = [
        "tao_bridge/xla_activity.proto",
    ],
)

cc_library(
    name = "tao_bridge",
    srcs = glob(
        include = [
            "tao_bridge/**/*.cc"
        ],
        exclude = [
            "tao_bridge/**/*test.cc",
            "tao_bridge/ral/**/*.cc",
        ],
    ),
    hdrs = glob(
        include = [
            "tao_bridge/**/*.h"
        ],
        exclude = [
            "tao_bridge/ral/**/*.h",
        ],
    ),
    deps = [
        ":tao_compiler_input",
        ":tao_compilation_result",
        ":xla_activity",
        "@local_config_tf//:tf_header_lib",
        "@local_config_tf//:libtensorflow_framework",
        "@nlohmann_json_lib//:nlohmann_json",
        "//tao/tao_bridge/ral:bridge_ral",
    ],
    copts = if_cuda([
        "-DGOOGLE_CUDA=1"
    ]) + tf_copts(),
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
    linkstatic = 1,
)
