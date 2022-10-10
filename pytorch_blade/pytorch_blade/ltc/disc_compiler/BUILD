load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("//bazel:build_defs.bzl", "if_ltc_disc_backend")

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
    "//pytorch_blade/compiler/mlir:torch_blade_mlir",
    "@local_org_torch//:ATen",
    "@local_org_torch//:libtorch", 
  ],
  alwayslink = True,
)

cc_library (
  name = "disc_compiler",
  srcs = [
    "disc_compiler.cpp",
  ] + if_ltc_disc_backend([
    "replay.cpp"
  ]),
  hdrs = [
    "disc_compiler.h",
  ] + if_ltc_disc_backend([
    "replay.h"
  ]),
  copts = select({
       "//:enable_cuda": ["-DTORCH_BLADE_BUILD_WITH_CUDA"],
       "//conditions:default": []}),
  includes = ["../include"],
  deps = [
    ":disc_passes",
    "@local_org_torch//:ATen",
    "@local_org_torch//:libtorch",
  ],
  visibility = [
    "//visibility:public",
  ],
)

cc_test(
  name = "ltc_disc_test",
  srcs = []
  + if_ltc_disc_backend([
    "disc_compiler_test.cpp",
    #"passes/disc_fuser_test.cpp",
  ]),
  deps = [
    "@local_org_torch//:libtorch",
    "@googltest//:gtest_main",
  ] + if_ltc_disc_backend([
    #TODO(yancey.yx): fix core dump while linking :torch_blade_mlir
    #":disc_passes",
  ])
)
