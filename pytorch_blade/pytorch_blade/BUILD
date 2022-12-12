load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda_is_configured",
)
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_import", "cc_library")
load(
    "//bazel:build_defs.bzl",
    "if_platform_alibaba",
    "if_tensorrt_enabled",
    "if_ltc_disc_backend",
    "if_quantization_enabled",
    "if_neural_engine_enabled"
)
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

package(default_visibility = ["//visibility:public"])

exports_files([
    "pybind.h",
])

cc_library(
    name = "torch_blade",
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    deps = [
        "//pytorch_blade/common_utils:torch_blade_common",
        "//pytorch_blade/compiler/backends:torch_blade_backends",
        "//pytorch_blade/compiler/jit:aten_custom_ops",
        "//pytorch_blade/compiler/jit:torch_blade_jit",
        "//pytorch_blade/compiler/mlir:torch_blade_mlir",
        "@local_org_torch//:libtorch",
    ] + if_tensorrt_enabled([
        "//pytorch_blade/compiler/tensorrt:tensorrt_runtime_impl",
    ]) + if_cuda_is_configured([
        "@local_config_cuda//cuda:cudart",
    ]) + if_platform_alibaba([
        "//pytorch_blade/platform_alibaba:torch_blade",
    ]) + if_quantization_enabled([
        "//pytorch_blade/quantization:quantization_op",
    ]) + if_neural_engine_enabled([
        "//pytorch_blade/compiler/neural_engine:torch_blade_neural_engine",
    ]),
)

cc_binary(
    name = "libtorch_blade.so",
    linkopts = [
        "-Wl,--version-script,$(location :torch_blade.lds)",
    ],
    linkshared = 1,
    deps = [
        ":torch_blade",
        ":torch_blade.lds",
    ],
)

cc_import(
    name = "torch_blade_shared",
    shared_library = "libtorch_blade.so",
)

# Pybind11 bindings for TorchBlade

licenses(["notice"])

exports_files(["LICENSE"])

filegroup(
    name = "torch_blade_py_sources",
    srcs = glob([
        "*pybind*.cpp",
    ]) + [
        "//pytorch_blade/compiler:torch_blade_py_srcs",
    ] + if_ltc_disc_backend([
         "//pytorch_blade/ltc:torch_blade_ltc_py_srcs",
    ]) + if_quantization_enabled([
        "//pytorch_blade/quantization:torch_blade_quantization_py_srcs",
    ]),
)

filegroup(
    name = "torch_blade_py_headers",
    srcs = glob([
        "*.h",
    ]) + [
        "//pytorch_blade/compiler:torch_blade_py_hdrs",
    ] + if_ltc_disc_backend([
         "//pytorch_blade/ltc:torch_blade_ltc_py_hdrs",
    ]) + if_quantization_enabled([
        "//pytorch_blade/quantization:torch_blade_quantization_py_hdrs",
    ]),
)

pybind_library(
    name = "torch_blade_pybind11",
    srcs = [
        ":torch_blade_py_sources",
    ],
    hdrs = [
        ":torch_blade_py_headers",
    ],
    copts = [
        "-DTORCH_BLADE_BUILD_MLIR",
    ] + if_tensorrt_enabled([
        "-DTORCH_BLADE_BUILD_TENSORRT",
    ]) + if_platform_alibaba(
        ["-DTORCH_BLADE_PLATFORM_ALIBABA=true"],
        ["-DTORCH_BLADE_PLATFORM_ALIBABA=false"],
    ) + if_ltc_disc_backend([
        "-DTORCH_BLADE_ENABLE_LTC"
    ]) + if_neural_engine_enabled([
        "-DTORCH_BLADE_BUILD_NEURAL_ENGINE"
    ]),
    includes = ["."],
    linkstatic = 0,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":torch_blade_shared",
        "//pytorch_blade/compiler/jit:onnx_funcs",
        "@local_org_torch//:torch_python",
    ] + if_tensorrt_enabled([
        "@tensorrt//:nvinfer_headers",
    ]) + if_ltc_disc_backend([
        "//pytorch_blade/ltc:torch_disc_backend",
    ]),
    alwayslink = 1,
)

pybind_extension(
    name = "_torch_blade",
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    deps = [
        ":torch_blade_shared",
        ":torch_blade_pybind11",
    ] + if_platform_alibaba([
        "//pytorch_blade/platform_alibaba:torch_blade_pybind11",
    ]),
)

test_suite(
    name = "torch_blade_test_suite",
    tests = [
        "//pytorch_blade/common_utils:torch_blade_common_utils_test",
        "//pytorch_blade/compiler/jit:jit_test",
        "//pytorch_blade/ltc/disc_compiler:ltc_disc_test",
    ],
)
