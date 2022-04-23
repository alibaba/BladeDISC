load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

cc_library (
  name = "torch_disc_backend",
  srcs =  [
    "ltc/disc_backend/backend_impl.cpp",
  ],
  hdrs = [
    "ltc/disc_backend/backend_impl.h",
  ],
  deps = [
    "//src/ltc/disc_compiler:disc_compiler"
  ]
)

pybind_library (
  name = "torch_disc_pybind11",
  srcs = ["ltc/init_python_bindings.cpp"],
  hdrs = ["ltc/init_python_bindings.h"],
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
    "//src/ltc/disc_compiler:torch_disc_test"
  ]
)
