load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

cc_library (
  name = "torch_disc_backend",
  srcs =  [
    "disc_backend/backend_impl.cpp",
  ],
  hdrs = [
    "disc_backend/backend_impl.h",
  ],
  deps = [
    "//pytorch_blade/ltc/disc_compiler:disc_compiler",
  ],
  visibility = [
      "//visibility:public",
  ],
  alwayslink = True,
)

filegroup(
    name = "torch_blade_ltc_py_srcs",
    srcs = ["init_python_bindings.cpp"],
    visibility = [
      "//visibility:public",
    ],
)

filegroup(
    name = "torch_blade_ltc_py_hdrs",
    srcs = ["init_python_bindings.h"],
    visibility = [
      "//visibility:public",
    ],
)
