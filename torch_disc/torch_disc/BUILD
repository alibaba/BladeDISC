load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

pybind_library (
  name = "torch_disc_pybind11",
  srcs = ["csrc/init_python_bindings.cpp"],
  deps = [
    "@local_org_torch//:ltc_ts_backend",
    # TODO(yancey1989): depends on one module
    "@org_torch_blade//src/compiler/mlir/converters:torch_blade_mhlo_converter",
    "@org_torch_blade//src/compiler/mlir/converters/impl:torch_blade_mhlo_converter_impl",
    "@org_torch_blade//src/compiler/jit:torch_blade_jit",
  ]
)

pybind_extension(
    name = "_torch_disc",
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    deps = [
        ":torch_disc_pybind11",
    ],
)