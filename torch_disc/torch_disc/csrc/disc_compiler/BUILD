load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

cc_library (
  name = "disc_compiler",
  srcs = [
    "disc_compiler.cpp",
    "passes/cluster.cpp",
    "passes/register_disc_class.cpp",
    "passes/disc_class.cpp",
  ],
  hdrs = [
    "disc_compiler.h",
    "passes/io.h",
    "passes/cluster.h",
    "passes/register_disc_class.h",
    "passes/disc_class.h",
  ],
  deps = [
    "@local_org_torch//:ltc_ts_backend",
    "@org_torch_blade//src/compiler/mlir:torch_blade_mlir",
    "@local_org_torch//:ATen",
    "@local_org_torch//:libtorch",
  ],
  alwayslink = True,
  visibility = [
    "//visibility:public",
  ],
)

cc_test(
  name = "torch_disc_test",
  srcs = [
    "disc_compiler_test.cpp",
  ],
  linkopts = [
    "-lpython3.8"
  ],
  deps = [
    #":torch_disc_backend",
    #"@local_org_torch//:torch_python",
    "@local_org_torch//:libtorch",
    "@googltest//:gtest_main",
  ]
)