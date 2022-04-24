load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

cc_library (
  name = "disc_compiler",
  srcs = [
    "disc_compiler.cpp",
    "passes/cluster.cpp",
    "passes/register_disc_class.cpp",
  ],
  hdrs = [
    "disc_compiler.h",
    "passes/io.h",
    "passes/cluster.h",
    "passes/register_disc_class.h",
  ],
  includes = ["../include"],
  deps = [
    "//src:torch_blade_include",
    "//src:torch_blade_shared",
    "@local_org_torch//:ATen",
    "@local_org_torch//:libtorch",
  ],
  visibility = [
    "//visibility:public",
  ],
)

cc_test(
  name = "torch_disc_test",
  srcs = [
    "disc_compiler_test.cpp",
  ],
  deps = [
    ":disc_compiler",    
    "//src:torch_blade_shared",
    "@org_tensorflow//tensorflow/compiler/mlir/xla/ral:ral_base_context_lib",
    "@local_org_torch//:libtorch",
    "@googltest//:gtest_main",
  ]
)
