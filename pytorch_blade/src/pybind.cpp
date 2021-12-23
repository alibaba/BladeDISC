
#include "pybind.h"

#include <mutex>
#include "common_utils/version.h"
#include "compiler/jit/onnx_funcs.h"
#include "compiler/jit/pybind_functions.h"
#ifdef TORCH_BLADE_BUILD_MLIR
#include "compiler/mlir/pybind_functions.h"
#endif // TORCH_BLADE_BUILD_MLIR)

// this include resolve some pybind11 incompatible problem of torch data
// structures like Dict
#include "torch/csrc/jit/python/pybind_utils.h"

namespace torch {
namespace blade {

namespace py = pybind11;
namespace {
std::string pybind_version() {
#define STR_EXPAND(token) #token
#define STR(token) STR_EXPAND(token)
  return "v" STR(PYBIND11_VERSION_MAJOR) "." STR(
      PYBIND11_VERSION_MINOR) "." STR(PYBIND11_VERSION_PATCH);
#undef STR
#undef STR_EXPAND
}
} // anonymous namespace

template <>
void initModules<COMMUNITY_VERSION_ID>(py::module& m) {
  torch::blade::initToolsBindings(m);
  m.def(
      "jit_pass_onnx_constant_f64_to_f32",
      &torch::blade::CastDownAllConstantDoubleToFloat);
#ifdef TORCH_BLADE_BUILD_MLIR
  torch::blade::initMLIRBindings(m);
#endif // TORCH_BLADE_BUILD_MLIR)
}

PYBIND11_MODULE(_torch_blade, m) {
  m.def("pybind_version", &pybind_version, R"pbdoc(
        return pybind version
    )pbdoc");
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  initModules<TORCH_BLADE_PLATFORM_VERSION_ID>(m);
}

} // namespace blade
} // namespace torch
