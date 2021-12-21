#ifndef TENSORFLOW_COMPILER_MLIR_XLA_RAL_CUSTOM_LIB_CALL_DYNAMIC_SORT_IMPL_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_RAL_CUSTOM_LIB_CALL_DYNAMIC_SORT_IMPL_H_

#include "tensorflow/compiler/mlir/xla/ral/context/common_context_impl.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_helper.h"

namespace {

struct SortDescriptor {
  unsigned sort_length;
  unsigned batch;
  bool is_ascending;
};

SortDescriptor makeSortDescriptor(tao::ral::ExecutionContext* ctx,
                                  const unsigned rank, const int64_t* sizes,
                                  const bool ascending) {
  SortDescriptor desc;
  if (rank > 2) {
    ctx->signalError(tao::ral::Context::FAILURE,
                     "Dynamic Sort do not support rank > 2");
  }
  desc.batch = (rank == 1) ? 1 : static_cast<unsigned>(sizes[0]);
  desc.sort_length = (rank == 1) ? static_cast<unsigned>(sizes[0])
                                 : static_cast<unsigned>(sizes[1]);
  desc.is_ascending = ascending;
  return desc;
}

template <typename Dtype>
struct IsIntTyped {
  static const bool value = false;
};
template <>
struct IsIntTyped<int> {
  static const bool value = true;
};
template <>
struct IsIntTyped<unsigned> {
  static const bool value = true;
};

}  // namespace

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_RAL_CUSTOM_LIB_CALL_DYNAMIC_SORT_IMPL_H_
