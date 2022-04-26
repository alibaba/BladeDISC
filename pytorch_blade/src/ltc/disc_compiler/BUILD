load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

cc_library (
  name = "disc_passes",
  srcs = [
    "passes/disc_fuser.cpp",
    "passes/register_disc_class.cpp",
    "passes/graph_fuser.cpp",
  ],
  hdrs = [
    "passes/io.h",
    "passes/disc_fuser.h",
    "passes/register_disc_class.h",
    "passes/graph_fuser.h",
  ],
  deps = [
    "//src/compiler/mlir:torch_blade_mlir",
    "@local_org_torch//:ATen",
    "@local_org_torch//:libtorch", 
  ],
  alwayslink = True,
)

cc_library (
  name = "disc_compiler",
  srcs = [
    "disc_compiler.cpp",
  ],
  hdrs = [
    "disc_compiler.h",
  ],
  includes = ["../include"],
  deps = [
    ":disc_passes",
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
    "passes/disc_fuser_test.cpp",
  ],
  deps = [
    ":disc_passes",
    "@local_org_torch//:libtorch",
    "@googltest//:gtest_main",
  ]
)
