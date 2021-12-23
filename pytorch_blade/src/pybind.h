#pragma once

#include <pybind11/pybind11.h>
#include <functional>

namespace torch {
namespace blade {

template <int PLATFORM_VERSION_ID>
void initModules(pybind11::module&); //# {}

} // namespace blade
} // namespace torch
