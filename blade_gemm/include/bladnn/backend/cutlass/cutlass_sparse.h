
#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <unordered_map>

#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "cutlass/library/util.h"

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////
//                          Data Structures for SparseGemm Functional Maps
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tuple uniquely identifying Gemm functional behavior
struct SparseGemmFunctionalKey {
  Provider provider;
  GemmKind gemm_kind;
  NumericTypeID element_compute;
  NumericTypeID element_scalar;
  NumericTypeID element_A;
  LayoutTypeID layout_A;
  ComplexTransform transform_A;
  NumericTypeID element_B;
  LayoutTypeID layout_B;
  ComplexTransform transform_B;
  NumericTypeID element_C;
  NumericTypeID element_E;

  //
  // Methods
  //

  inline SparseGemmFunctionalKey(
      Provider provider, GemmKind gemm_kind = GemmKind::kGemm,
      NumericTypeID element_compute = NumericTypeID::kF32,
      NumericTypeID element_scalar = NumericTypeID::kF32,
      NumericTypeID element_A = NumericTypeID::kF16,
      LayoutTypeID layout_A = LayoutTypeID::kColumnMajor,
      ComplexTransform transform_A = ComplexTransform::kNone,
      NumericTypeID element_B = NumericTypeID::kF16,
      LayoutTypeID layout_B = LayoutTypeID::kColumnMajor,
      ComplexTransform transform_B = ComplexTransform::kNone,
      NumericTypeID element_C = NumericTypeID::kF16,
      NumericTypeID element_E = NumericTypeID::kS16)
      : provider(provider),
        gemm_kind(gemm_kind),
        element_compute(element_compute),
        element_scalar(element_scalar),
        element_A(element_A),
        layout_A(layout_A),
        transform_A(transform_A),
        element_B(element_B),
        layout_B(layout_B),
        transform_B(transform_B),
        element_C(element_C),
        element_E(element_E) {}

  inline bool operator==(SparseGemmFunctionalKey const& rhs) const {
    return (provider == rhs.provider) && (gemm_kind == rhs.gemm_kind) &&
           (element_compute == rhs.element_compute) &&
           (element_scalar == rhs.element_scalar) &&
           (element_A == rhs.element_A) && (layout_A == rhs.layout_A) &&
           (transform_A == rhs.transform_A) && (element_B == rhs.element_B) &&
           (layout_B == rhs.layout_B) && (transform_B == rhs.transform_B) &&
           (element_C == rhs.element_C) && (element_E == rhs.element_E);
  }

  inline bool operator!=(SparseGemmFunctionalKey const& rhs) const {
    return !(*this == rhs);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
inline std::ostream& operator<<(
    std::ostream& out, cutlass::library::SparseGemmFunctionalKey const& k) {
  out << "{\n"
      << "         provider: " << to_string(k.provider) << "\n"
      << "        gemm_kind: " << to_string(k.gemm_kind) << "\n"
      << "  element_compute: " << to_string(k.element_compute) << "\n"
      << "   element_scalar: " << to_string(k.element_scalar) << "\n"
      << "        element_A: " << to_string(k.element_A) << "\n"
      << "         layout_A: " << to_string(k.layout_A) << "\n"
      << "      transform_A: " << to_string(k.transform_A) << "\n"
      << "        element_B: " << to_string(k.element_B) << "\n"
      << "         layout_B: " << to_string(k.layout_B) << "\n"
      << "      transform_B: " << to_string(k.transform_B) << "\n"
      << "        element_C: " << to_string(k.element_C) << "\n"
      << "        element_E: " << to_string(k.element_E) << "\n"
      << "}";

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Hash function for GemmFunctionalKey
struct SparseGemmFunctionalKeyHasher {
  using IntHash = std::hash<int>;

  inline static size_t rotl(size_t key, int shl) {
    return (key << shl) | (key >> (sizeof(key) * 8 - shl));
  }

  inline size_t operator()(SparseGemmFunctionalKey const& key) const {
    IntHash hash;

    return rotl(hash(int(key.provider)), 1) ^
           rotl(hash(int(key.gemm_kind)), 2) ^
           rotl(hash(int(key.element_compute)), 3) ^
           rotl(hash(int(key.element_scalar)), 4) ^
           rotl(hash(int(key.element_A)), 5) ^
           rotl(hash(int(key.layout_A)), 6) ^
           rotl(hash(int(key.transform_A)), 7) ^
           rotl(hash(int(key.element_B)), 8) ^
           rotl(hash(int(key.layout_B)), 9) ^
           rotl(hash(int(key.transform_B)), 10) ^
           rotl(hash(int(key.element_C)), 11) ^
           rotl(hash(int(key.element_E)), 12);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Establishes a partial ordering to search for GEMM operators
struct SparseGemmPreferenceKey {
  int compute_capability;
  int alignment;

  //
  // Methods
  //

  SparseGemmPreferenceKey() : compute_capability(), alignment() {}

  SparseGemmPreferenceKey(int cc, int alignment)
      : compute_capability(cc), alignment(alignment) {}

  bool operator<(SparseGemmPreferenceKey const& rhs) const {
    return (compute_capability < rhs.compute_capability) ||
           ((compute_capability == rhs.compute_capability) &&
            (alignment < rhs.alignment));
  }

  bool operator==(SparseGemmPreferenceKey const& rhs) const {
    return compute_capability == rhs.compute_capability;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Maps minimum compute capability onto a vector of possible operations
using SparseGemmOperationVectorMap =
    std::map<SparseGemmPreferenceKey, std::vector<Operation const*> >;

/// Maps a GemmFunctionalKey onto a vector of Operation * objects expected to be
/// of kind kGemm
using SparseGemmOperationFunctionalMap =
    std::unordered_map<SparseGemmFunctionalKey, SparseGemmOperationVectorMap,
                       SparseGemmFunctionalKeyHasher>;
/////////////////////////////////////////////////////////////////////////////////////////////////

int spgemm_problem_alignment(
    int M, int N, int K, NumericTypeID element_A, void const* ptr_A,
    int64_t lda, int64_t batch_stride_A, NumericTypeID element_B,
    void const* ptr_B, int64_t ldb, int64_t batch_stride_B,
    NumericTypeID element_C, void const* ptr_C, int64_t ldc,
    int64_t batch_stride_C, void const* ptr_D, int64_t ldd,
    int64_t batch_stride_D, NumericTypeID element_E, void const* ptr_E,
    int64_t lde, int64_t batch_stride_E, int max_alignment_in_bytes = 16);

class SparseOperationTable {
 public:
  /// Manifest object
  Manifest manifest;

  /// Map of all operations of type kSpGemm
  // provider (kCUTLASS)
  SparseGemmOperationFunctionalMap sparse_gemm_operations;

 public:
  SparseOperationTable();

  void append(Manifest const& manifest);
};

}  // namespace library
}  // namespace cutlass