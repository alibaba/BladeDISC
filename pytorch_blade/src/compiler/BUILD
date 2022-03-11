package(default_visibility = ["//visibility:public"])

filegroup(
    name = "torch_blade_py_srcs",
    srcs = [
        "//src/compiler/jit:torch_blade_jit_py_srcs",
        "//src/compiler/mlir:torch_blade_mlir_py_srcs",
    ] + select({
        "//conditions:default": [],
        "//:enable_tensorrt":["//src/compiler/tensorrt:pybind_functions.cpp"]}),
)

filegroup(
    name = "torch_blade_py_hdrs",
    srcs = [
        "//src/compiler/jit:torch_blade_jit_py_hdrs",
        "//src/compiler/mlir:torch_blade_mlir_py_hdrs",
    ] + select({
        "//conditions:default": [],
        "//:enable_tensorrt":["//src/compiler/tensorrt:pybind_functions.h"]}),
)
