#ifndef __COMPILER_MLIR_PYBIND_H__
#define __COMPILER_MLIR_PYBIND_H__

#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace torch {
namespace addons {

void initMLIRBindings(py::module& m);
}
} // namespace torch
#endif //__COMPILER_MLIR_PYBIND_H__
