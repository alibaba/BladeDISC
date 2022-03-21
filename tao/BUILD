package(default_visibility = ["//visibility:public"])
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@local_config_tf//:build_defs.bzl", "tf_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load("//:build_defs.bzl", "tao_bridge_cc_test")

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
            "tao_bridge/gpu/bace_gemm_thunk.cc",
        ],
    ),
    hdrs = glob(
        include = [
            "tao_bridge/**/*.h"
        ],
        exclude = [
            "tao_bridge/ral/**/*.h",
            "tao_bridge/gpu/bace_gemm_thunk.h",
        ],
    ) + [
        "third_party/json/single_include/nlohmann/json.hpp",
    ],
    deps = [
        ":tao_compiler_input",
        ":tao_compilation_result",
        ":xla_activity",
        "@local_config_tf//:tf_header_lib",
        "@local_config_tf//:libtensorflow_framework",
        "@nlohmann_json_lib//:nlohmann_json",
        "//tao_bridge/ral:bridge_ral",
    ] + if_cuda([
        "//tao_bridge/ral:bridge_ral_cuda",
    ]),
    includes = [
        ".",
        "tao_bridge/ral",
    ],
    copts = if_cuda([
        "-DGOOGLE_CUDA=1",
    ]) + tf_copts(),
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
    linkstatic = 1,
)

cc_test(
    name = "tao_bridge_tests",
    srcs = [
        "tao_bridge/common_test.cc",
        "tao_bridge/cuda_utils_test.cc",
        "tao_bridge/dumper_common_test.cc",
        "tao_bridge/errors_test.cc",
        "tao_bridge/tao_util_test.cc",
    ],
    linkopts = [
        "-lpthread",
        "-lm",
        "-ldl",
    ],
    deps = [
        ":tao_bridge",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:tf_header_lib",
        "@local_config_tf//:libtensorflow_framework",
    ],
)

tao_bridge_cc_test(
    name = "tao_port_tests",
    srcs = [
        "tao_bridge/tf/const_analysis_test.cc",
        "tao_bridge/tf/deadness_analysis_test.cc",
        "tao_bridge/tf/graphcycles_test.cc",
        "tao_bridge/tf/resource_operation_safety_analysis_test.cc",
        "tao_bridge/tf/resource_operation_table_test.cc",
        "tao_bridge/tf/util_test.cc",
        "tao_bridge/tf/xla_cluster_util_test.cc",
        "tao_bridge/tf/xla_op_registry_test.cc",
    ],
)

tao_bridge_cc_test(
    name = "tao_passes_tests",
    srcs = [
        "tao_bridge/passes/tao_bace_reformat_pass_test.cc",
        "tao_bridge/passes/tao_encapsulate_subgraphs_pass_test.cc",
        # "tao_bridge/passes/tao_partially_decluster_pass_test.cc",
    ],
)

tao_bridge_cc_test(
    name = "tao_cpu_passes_tests",
    srcs = [
        "tao_bridge/passes/tao_clone_constants_for_better_clustering_test.cc",
        "tao_bridge/passes/tao_defuse_pass_test.cc",
        "tao_bridge/passes/tao_remove_small_cluster_pass_test.cc",
    ],
)
