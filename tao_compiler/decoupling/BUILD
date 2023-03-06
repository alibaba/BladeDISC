licenses(["notice"])  # Apache 2.0

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load("//mlir/util:util.bzl", "if_platform_alibaba")

package_group(
    name = "internal",
    packages = [
        "//tensorflow/compiler/aot/...",
        "//tensorflow/compiler/jit/...",
        "//tensorflow/compiler/tests/...",
        "//tensorflow/compiler/tf2xla/...",
    ],
)

package_group(
    name = "friends",
    includes = [":internal"],
    packages = ["//tensorflow/..."],
)

package(
    default_visibility = [":internal"],
)

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("//mlir/util:util.bzl", "if_cuda_or_rocm")
load(
    "@com_google_protobuf//:protobuf.bzl",
    "cc_proto_library",
)

cc_proto_library(
    name = "tao_compiler_input",
    srcs = ["tao_compiler_input.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_proto_library(
    name = "tao_compilation_result",
    srcs = ["tao_compilation_result.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tao_compiler_trace",
    srcs = ["tao_compiler_trace.cc"],
    hdrs = ["tao_compiler_trace.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "tao_compiler_trace_test",
    size = "small",
    srcs = ["tao_compiler_trace_test.cc"],
    linkstatic = 1,
    deps = [
        "tao_compiler_trace",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "disc_custom_ops",
    srcs = ["fake_quant_op.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework"
    ],
    alwayslink = 1,
)

cc_library(
    name = "compiler_base",
    srcs = ["compiler_base.cc"],
    hdrs = ["compiler_base.h"],
    visibility = ["//visibility:public"],
    deps = [
        "tao_compiler_input",
        "tao_compilation_result",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
    ]
)

cc_library(
    name = "tao_compiler",
    srcs = [
        "mlir_compiler.cc",
        "mlir_compiler_impl_cpu.cc",
    ] + if_cuda_is_configured([
        "mlir_compiler_impl_gpu.cc",
    ]) + if_rocm_is_configured([
        "mlir_compiler_impl_rocm.cc",
    ]),
    hdrs = [
        "mlir_compiler.h",
        "mlir_compiler_impl_cpu.h",
    ] + if_cuda_is_configured([
        "mlir_compiler_impl_gpu.h",
    ]) + if_rocm_is_configured([
        "mlir_compiler_impl_rocm.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":compiler_base",
        ":disc_custom_ops",
        ":tao_compiler_trace",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/cc:resource_variable_ops",
        "@org_tensorflow//tensorflow/cc:function_ops",
        "@org_tensorflow//tensorflow/compiler/jit:jit",
        "@org_tensorflow//tensorflow/compiler/tf2xla:tf2xla",
        "@org_tensorflow//tensorflow/compiler/xla:error_spec",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:literal_comparison",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla:status_macros",
        "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
        "@org_tensorflow//tensorflow/compiler/xla/client:local_client",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
        "@org_tensorflow//tensorflow/compiler/xla/hlo/ir:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_runner",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:core_cpu_internal",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:framework_internal",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:lib_internal",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ] + if_cuda_or_rocm([
        "@org_tensorflow//tensorflow/compiler/xla/service/gpu:gpu_transfer_manager",
    ]) + if_rocm_is_configured([
        "@org_tensorflow//tensorflow/stream_executor/rocm:rocm_driver",
    ]) + [
        "@llvm-project//mlir:IR",
        "@llvm-project//llvm:Support",
        "//mlir/disc:disc_util",
        "//mlir/disc:disc_compiler",
    ],
    alwayslink = True,
)

tf_cc_binary(
    name = "tao_compiler_main",
    srcs = [
        "tao_compiler_main.cc",
        "version.h",
    ],
    linkopts = ["-fno-lto -lutil -export-dynamic"],
    linkstatic = 1,
    deps = [
        "tao_compiler",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span"
    ],
)

tf_cc_binary(
    name = "tao_test_runner",
    srcs = [
        "tao_test_runner.cc",
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/tf2xla",
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_comparison",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service/gpu:gpu_transfer_manager",
        "//tensorflow/compiler/xla/service/gpu:tao_compiler",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_binary(
    name = "tao_grappler_transformer",
    srcs = [
        "tao_grappler_transformer.cc",
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "tao_compiler_test",
    srcs = [
        "tao_compiler_test.cc",
    ],
    data = [
        "tests_data/mlir_gpu.proto",
    ],
    linkstatic = 1,
    deps = [
        ":tao_compilation_result",
        ":tao_compiler",
        ":tao_compiler_input",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
