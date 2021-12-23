#ifndef __JIT_PYBIND_FUNCS_H__
#define __JIT_PYBIND_FUNCS_H__

#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace torch {
namespace blade {

void initToolsBindings(py::module& m);
}
} // namespace torch
#endif //__JIT_PYBIND_FUNCS_H__
