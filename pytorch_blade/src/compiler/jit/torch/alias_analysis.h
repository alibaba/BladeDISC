#ifndef __COMPILER_ALIAS_ANALYSIS_H__
#define __COMPILER_ALIAS_ANALYSIS_H__
#include <torch/script.h>

namespace torch {
namespace blade {
// TODO: These two methods were added because
// torch::jit::isMutableType is not externally visiable
bool isMutableType(const torch::jit::TypePtr& type);
bool isMutableType(const torch::jit::Value* v);
} // namespace blade
} // namespace torch

#endif //__COMPILER_ALIAS_ANALYSIS_H__
