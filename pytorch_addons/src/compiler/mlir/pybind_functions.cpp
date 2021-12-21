#include "pybind.h"

#include "compiler/mlir/converters/mhlo_conversion.h"
#include "compiler/mlir/pybind_functions.h"
#include "compiler/mlir/runtime/disc_engine_class.h"

#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace addons {

void initMLIRBindings(py::module& m) {
  py::module mlir =
      m.def_submodule("_mlir", "torch_addons python bindings to mlir");
  mlir.def("cvt_torchscript_to_mhlo", &ConvertTorchScriptToMhlo);
  mlir.def("is_mlir_mhlo_supported", &IsMlirMhloSupported);
  mlir.def("register_disc_engine", &register_disc_engine);
}

} // namespace addons
} // namespace torch
