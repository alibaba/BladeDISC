load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

cc_library (
  name = "torch_disc_backend",
  srcs =  [
    "csrc/disc_backend/backend_impl.cpp",
  ],
  hdrs = [
    "csrc/disc_backend/backend_impl.h",
  ],
  deps = [
    "@local_org_torch//:ltc_ts_backend",
    "//torch_disc/csrc/disc_compiler:disc_compiler"
  ]
)
#cc_library (
#  name = "torch_disc_backend",
#  srcs = [
#    "csrc/backend_impl.cpp",
#    "csrc/disc_jit.cpp",
#    "csrc/disc_class.cpp",
#  ],
#  hdrs = [
#    "csrc/backend_impl.h",
#    "csrc/disc_jit.h",
#    "csrc/disc_class.h",
#    "csrc/io.h",
#  ],
#  deps = [
#    "@local_org_torch//:ltc_ts_backend",
#    # TODO(yancey1989): depends on one module
#    "@org_torch_blade//src/compiler/mlir/converters:torch_blade_mhlo_converter",
#    "@org_torch_blade//src/compiler/mlir/converters/impl:torch_blade_mhlo_converter_impl",
#    "@org_torch_blade//src/compiler/jit:torch_blade_jit",
#    "@org_torch_blade//src/compiler/mlir/runtime:torch_blade_ral_context",
#    "@local_org_torch//:ATen",
#    "@local_org_torch//:libtorch",
#  ],
#  alwayslink = True,
#)

pybind_library (
  name = "torch_disc_pybind11",
  srcs = ["csrc/init_python_bindings.cpp"],
  deps = [
    ":torch_disc_backend",
  ]
)

pybind_extension(
  name = "_torch_disc",
  linkopts = ["-Wl,-rpath,$$ORIGIN"],
  deps = [
    ":torch_disc_pybind11",
  ],
)



test_suite(
  name = "torch_disc_test_suit",
  tests = [
    "//torch_disc/csrc/disc_compiler:torch_disc_test"
  ]
)
