load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

cc_library (
  name = "torch_disc_backend",
  srcs =  [
    "disc_backend/backend_impl.cpp",
  ],
  hdrs = [
    "disc_backend/backend_impl.h",
  ],
  includes = ["include"],
  deps = [
    "//src/compiler/ltc/disc_compiler:disc_compiler"
  ],
)

pybind_library (
  name = "torch_disc_pybind11",
  srcs = ["init_python_bindings.cpp"],
  deps = [
    ":torch_disc_backend",
  ],
  visibility = [
      "//visibility:public",
  ],
)

pybind_extension(
  name = "_torch_disc",
  linkopts = ["-Wl,-rpath,$$ORIGIN"],
  deps = [
    ":torch_disc_pybind11",
    "@local_org_torch//:torch_python",
  ],
)

# test_suite(
#    name = "torch_disc_test_suit",
#    tests = [
#      "//src/compiler/ltc/disc_compiler:torch_disc_test"
#    ]
# )
