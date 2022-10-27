#include "bladnn/backend/cutlass/cutlass_sparse.h"

#include <iostream>

#include "cutlass/library/operation_table.h"

namespace cutlass {
namespace library {

int spgemm_problem_alignment(
    int M, int N, int K, NumericTypeID element_A, void const* ptr_A,
    int64_t lda, int64_t batch_stride_A, NumericTypeID element_B,
    void const* ptr_B, int64_t ldb, int64_t batch_stride_B,
    NumericTypeID element_C, void const* ptr_C, int64_t ldc,
    int64_t batch_stride_C, void const* ptr_D, int64_t ldd,
    int64_t batch_stride_D, NumericTypeID element_E, void const* ptr_E,
    int64_t lde, int64_t batch_stride_E, int max_alignment_in_bytes) {
  void const* pointers[] = {ptr_A, ptr_B, ptr_C, ptr_D, ptr_E};

  int64_t extents[] = {M,
                       N,
                       K,
                       lda,
                       ldb,
                       ldc,
                       ldd,
                       lde,
                       batch_stride_A,
                       batch_stride_B,
                       batch_stride_C,
                       batch_stride_D,
                       batch_stride_E};

  NumericTypeID elements[] = {element_A, element_B, element_C, element_E};

  for (; max_alignment_in_bytes > 0; max_alignment_in_bytes /= 2) {
    bool satisfied = true;

    // Can pointers satisfy this?
    for (void const* ptr : pointers) {
      std::uintptr_t int_ptr = reinterpret_cast<std::uintptr_t>(ptr);

      if (int_ptr % max_alignment_in_bytes) {
        satisfied = false;
        break;
      }
    }

    if (!satisfied) {
      continue;
    }

    // Compute the maximum alignment based on element data types
    int max_element_alignment = 0;

    for (NumericTypeID type_id : elements) {
      int element_alignment =
          max_alignment_in_bytes * 8 / library::sizeof_bits(type_id);
      max_element_alignment =
          std::max(max_element_alignment, element_alignment);
    }

    // Can the problem size and leading dimensions satisfy this?
    for (int64_t extent : extents) {
      if (extent % max_element_alignment) {
        satisfied = false;
        break;
      }
    }

    if (!satisfied) {
      continue;
    }

    // Yes
    return max_element_alignment;
  }

  // No alignment satisfies this problem
  return 0;
}

SparseOperationTable::SparseOperationTable() {
  manifest.initialize();
  this->append(manifest);
}

void SparseOperationTable::append(Manifest const& manifest) {
  // Insert operations into appropriate data structure
  for (auto const& operation : manifest) {
    OperationDescription const& desc = operation->description();

    if (desc.kind == OperationKind::kSparseGemm) {
      SparseGemmDescription const& spgemm_desc =
          static_cast<SparseGemmDescription const&>(desc);

      SparseGemmFunctionalKey functional_key(
          spgemm_desc.provider, spgemm_desc.gemm_kind,
          spgemm_desc.tile_description.math_instruction.element_accumulator,
          spgemm_desc.element_epilogue, spgemm_desc.A.element,
          spgemm_desc.A.layout, spgemm_desc.transform_A, spgemm_desc.B.element,
          spgemm_desc.B.layout, spgemm_desc.transform_B, spgemm_desc.C.element,
          spgemm_desc.E.element);

      Operation const* op = operation.get();

      int cc = spgemm_desc.tile_description.minimum_compute_capability;

      int alignment =
          std::max(std::max(spgemm_desc.A.alignment, spgemm_desc.B.alignment),
                   spgemm_desc.C.alignment);

      SparseGemmPreferenceKey preference_key(cc, alignment);

      sparse_gemm_operations[functional_key][preference_key].push_back(op);
    }
  }
}

}  // namespace library
}  // namespace cutlass