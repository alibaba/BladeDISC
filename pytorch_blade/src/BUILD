load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda_is_configured",
)

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
cc_library(
    name = "torch_blade",
    deps = [
        "//src/common_utils:torch_blade_common",
        "//src/compiler/jit:torch_blade_jit",
        "//src/compiler/mlir:torch_blade_mlir",
        "@local_org_torch//:libtorch",
        "@org_mhlo_builder//csrc:mhlo_library",
        "@org_tensorflow//tensorflow/compiler/mlir/disc:mhlo_disc",
        "@org_tensorflow//tensorflow/compiler/mlir/disc:mlir_disc_headers_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/disc:mlir_disc_builder.so",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo:lhlo",
        "@org_tensorflow//tensorflow/compiler/mlir/xla/ral:ral_base_context_lib",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cudart",])
)

# Pybind11 bindings for TorchBlade

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

licenses(["notice"])

exports_files(["LICENSE"])

filegroup(
    name = "torch_blade_py_sources",
    srcs = glob(["*pybind*.cpp"]) + ["//src/compiler:torch_blade_py_srcs"],
)

filegroup(
    name = "torch_blade_py_headers",
    srcs = glob(["*.h"]) + ["//src/compiler:torch_blade_py_hdrs"],
)

pybind_library(
    name = "torch_blade_pybind11",
    srcs = [":torch_blade_py_sources"],
    hdrs = [
        ":torch_blade_py_headers",
    ],
    includes = ["."],
    deps = [
       ":torch_blade",
       "@local_org_torch//:torch_python",
    ],
    visibility = [
        "//visibility:public",
    ],
    linkstatic = 0,
)

pybind_extension(
    name = "_torch_blade",
    deps = [
      ":torch_blade_pybind11",],
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
)

test_suite(
    name = "torch_blade_gtests",
    tests = [
        "//src/compiler/jit:shape_type_test",
        "//src/compiler/mlir/converters:mhlo_converter_register_test",
    ],
)
