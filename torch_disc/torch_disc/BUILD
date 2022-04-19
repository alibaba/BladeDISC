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
