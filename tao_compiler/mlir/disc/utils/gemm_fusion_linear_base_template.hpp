R"bladedisc_rstr(

class SpecializedGemmFusion {
 public:
  template <typename T>
  struct ElementwiseUnaryOp {
    __inline__ __attribute__((always_inline)) __attribute__((device))
    T operator()(T const& input) const {
      return input;
    }
  };

  template <typename T>
  struct EpilogueFunctor {
    static const bool kIsHeavy = EpilogueIsHeavy;

    __inline__ __attribute__((always_inline)) __attribute__((device))
    T operator()(T const& scalar) const {
      ElementwiseUnaryOp<T> op;

      return op(scalar);
    }

    using Params =
        cutlass::epilogue::thread::LinearCombinationGenericParams<T>;

    __inline__ __attribute__((always_inline)) __attribute__((device))
    T operator()(T const& scalar, Params const& params_) const {
      return this->operator()(scalar);
    }
  };

  template <typename T, int N>
  struct EpilogueFunctor<cutlass::Array<T, N>> {
    static const bool kIsHeavy = EpilogueIsHeavy;

    __inline__ __attribute__((always_inline)) __attribute__((device))
    cutlass::Array<T, N> operator()(cutlass::Array<T, N>
                                    const& frag) const {
      cutlass::Array<T, N> result;
      ElementwiseUnaryOp<T> op;

      #pragma unroll
      for (int i = 0; i < N; ++i) {
        result[i] = op(frag[i]);
      }

      return result;
    }

    using Params =
        cutlass::epilogue::thread::LinearCombinationGenericParams<T>;

    __inline__ __attribute__((always_inline)) __attribute__((device))
    cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& frag,
                                    Params const& params_) const {
      return this->operator()(frag);
    }
  };

 public:
  SpecializedGemmFusion(void* stream, int64_t batch_size, int64_t m,
                        int64_t n, int64_t k, const void* A,
                        const void* B, void* D)
      : stream_(stream),
        batch_size_(batch_size),
        m_(m),
        n_(n),
        k_(k),
        A_(A),
        B_(B),
        D_(D) {}
  bool run();

 private:
  void* stream_;
  int64_t batch_size_;
  int64_t m_;
  int64_t n_;
  int64_t k_;
  const void* A_;
  const void* B_;
  void* D_;
};

template <>
struct SpecializedGemmFusion::ElementwiseUnaryOp<EpilogueElementType> {
  __inline__ __attribute__((always_inline)) __attribute__((device))
  EpilogueElementType operator()(
      EpilogueElementType const& input) const {
SpecializedEpilogue
  }
};

bool SpecializedGemmFusion::run() {
  bool debug_input = true;
  if (debug_input) {
    ElementAType* A_host = (ElementAType*)malloc(sizeof(ElementAType) * batch_size_ * m_ * k_);
    ElementAType* B_host = (ElementBType*)malloc(sizeof(ElementBType) * batch_size_ * k_ * n_);
    cudaMemcpy(A_host, A_, sizeof(ElementAType) * batch_size_ * m_ * k_, cudaMemcpyDefault);
    cudaMemcpy(B_host, B_, sizeof(ElementBType) * batch_size_ * k_ * n_, cudaMemcpyDefault);
    for (int b = 0; b < 2; b++) {
      for (int m = 0; m < 4; m++) {
        for (int k = 0; k < 4; k++) {
          std::cout << "A val at " << b << "," << m << "," << k << ": "
                    << A_host[b * m_ * k_ + m * k_ + k] << std::endl;
        }
      }
    }
    for (int b = 0; b < 2; b++) {
      for (int k = 0; k < 4; k++) {
        for (int n = 0; n < 4; n++) {
          std::cout << "B val at " << b << "," << k << "," << n <<": "
                    << B_host[b * k_ * n_ + k * n_ + n] << std::endl;
        }
      }
    }
  }

  constexpr cutlass::FloatRoundStyle Round =
      cutlass::FloatRoundStyle::round_to_nearest;
  constexpr cutlass::ComplexTransform TransformA =
      cutlass::ComplexTransform::kNone;
  constexpr cutlass::ComplexTransform TransformB =
      cutlass::ComplexTransform::kNone;

  using ElementA = ElementAType;
  using LayoutA = ElementALayout;
  using ElementB = ElementBType;
  using LayoutB = ElementBLayout;
  using ElementOutput = ElementOutputType;
  using LayoutOutput = ElementOutputLayout;
  using ElementAccumulator = ElementAccumulatorType;
  using OperatorClass = OperatorClassType;
  using ArchTag = SMArch;

  constexpr cutlass::epilogue::thread::ScaleType::Kind Scale =
      EpilogueScaleKind;
  constexpr bool IsHeavy = EpilogueIsHeavy;
  constexpr int Count = EpilogueCountVectorized;
  using ElementComputeEpilogue = EpilogueElementType;
  using EpilogueOutputOp =
      cutlass::epilogue::thread::LinearCombinationGeneric<
          EpilogueFunctor, ElementOutput, Count, ElementAccumulator,
          ElementComputeEpilogue, Scale, Round, IsHeavy>;

  constexpr bool GatherA = IsGatherA;
  constexpr bool GatherB = IsGatherB;
  constexpr bool ScatterD = IsScatterD;
  using PermuteDLayout = EpiloguePermuteDLayout;

  cutlass::gemm::GemmUniversalMode mode;
  if (batch_size_ > 1) {
    mode = cutlass::gemm::GemmUniversalMode::kBatched;
  } else {
    if (true) {
      mode = cutlass::gemm::GemmUniversalMode::kGemm;
    }
  }

  cutlass::gemm::GemmCoord problem_size(m_, n_, k_);

  typename EpilogueOutputOp::Params epilogue(
      ElementComputeEpilogue(1), ElementComputeEpilogue(0));

  long long int batch_stride_A =
      static_cast<long long int>(m_) * static_cast<long long int>(k_);
  long long int batch_stride_B =
      static_cast<long long int>(k_) * static_cast<long long int>(n_);
  long long int batch_stride_C = 0;
  long long int batch_stride_D =
      static_cast<long long int>(m_) * static_cast<long long int>(n_);

  int const lda = std::is_same<LayoutA, cutlass::layout::ColumnMajor>::value ? m_ : k_;
  int const ldb = std::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? k_ : n_;
  int const ldc = 0;
  int const ldd = n_;

  int const* ptr_gather_A_indices = nullptr;
  int const* ptr_gather_B_indices = nullptr;
  int const* ptr_scatter_D_indices = nullptr;

  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  int i_am_comment_the_following_rules_will_be_enriched;
  if (std::is_same<ElementA, cutlass::half_t>::value &&
      std::is_same<ElementB, cutlass::half_t>::value &&
      std::is_same<ElementOutput, cutlass::half_t>::value) {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using ThreadblockSwizzle =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
    constexpr int Stages = 3;

    using GemmConfiguration =
        cutlass::gemm::device::DefaultGemmConfiguration<
            OperatorClass, ArchTag, cutlass::half_t, cutlass::half_t, cutlass::half_t,
            ElementAccumulator>;
    constexpr int AlignmentA = GemmConfiguration::kAlignmentA;
    constexpr int AlignmentB = GemmConfiguration::kAlignmentB;
    using Operator = GemmConfiguration::Operator;

    using Gemm = cutlass::gemm::device::GemmUniversal<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, cutlass::half_t,
        LayoutOutput, ElementAccumulator, OperatorClass, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, AlignmentA,
        AlignmentB, Operator, TransformA, TransformB, GatherA, GatherB,
        ScatterD, PermuteDLayout>;

    typename Gemm::Arguments arguments{mode,
                                       problem_size,
                                       static_cast<int>(batch_size_),
                                       epilogue,
                                       A_,
                                       B_,
                                       nullptr,
                                       D_,
                                       batch_stride_A,
                                       batch_stride_B,
                                       batch_stride_C,
                                       batch_stride_D,
                                       lda,
                                       ldb,
                                       ldc,
                                       ldd,
                                       ptr_gather_A_indices,
                                       ptr_gather_B_indices,
                                       ptr_scatter_D_indices};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t>
        workspace(workspace_size);

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      return false;
    }
  } else if (std::is_same<ElementA, float>::value &&
      std::is_same<ElementB, float>::value &&
      std::is_same<ElementOutput, float>::value) {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    using ThreadblockSwizzle =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
    constexpr int Stages = 3;

    using GemmConfiguration =
        cutlass::gemm::device::DefaultGemmConfiguration<
            OperatorClass, ArchTag, float, float, float,
            ElementAccumulator>;
    constexpr int AlignmentA = GemmConfiguration::kAlignmentA;
    constexpr int AlignmentB = GemmConfiguration::kAlignmentB;
    using Operator = GemmConfiguration::Operator;

    using Gemm = cutlass::gemm::device::GemmUniversal<
        float, LayoutA, float, LayoutB, float,
        LayoutOutput, ElementAccumulator, OperatorClass, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, AlignmentA,
        AlignmentB, Operator, TransformA, TransformB, GatherA, GatherB,
        ScatterD, PermuteDLayout>;

    typename Gemm::Arguments arguments{mode,
                                       problem_size,
                                       static_cast<int>(batch_size_),
                                       epilogue,
                                       A_,
                                       B_,
                                       nullptr,
                                       D_,
                                       batch_stride_A,
                                       batch_stride_B,
                                       batch_stride_C,
                                       batch_stride_D,
                                       lda,
                                       ldb,
                                       ldc,
                                       ldd,
                                       ptr_gather_A_indices,
                                       ptr_gather_B_indices,
                                       ptr_scatter_D_indices};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t>
        workspace(workspace_size);

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      return false;
    }
  } else {
    return false;
  }

  return true;
}

extern "C"
bool gemmFusionFunc(void* stream, void** params) {

  int64_t size_struct = 3 + GRank * 2;
  int64_t batch_size = 1;
  for (int64_t i = 0; i < GRank - 2; ++i) {
    int64_t* a_size = reinterpret_cast<int64_t*>(params[3 + i]);
    batch_size *= *a_size;
  }

  int param_permute[] = ParameterPermute;
  if (true) {
    std::cout << "[ZZ] param permute:" << param_permute[0] << ", "
              << param_permute[1] << ", " << param_permute[2] << std::endl;
    std::cout << "[ZZ] batch_size: " << batch_size << std::endl;
    std::cout << "[ZZ] A sizes.\n";
    for (int i = 0; i < GRank; i++) {
      int64_t* a_size = reinterpret_cast<int64_t*>(params[3 + i + param_permute[0] * size_struct]);
      std::cout << i << ":" << *a_size << "\n";
    }
    std::cout << "[ZZ] B sizes.\n";
    for (int i = 0; i < GRank; i++) {
      int64_t* b_size = reinterpret_cast<int64_t*>(params[3 + i + param_permute[1] * size_struct]);
      std::cout << i << ":" << *b_size << "\n";
    }
    std::cout << std::flush;
  }

  int64_t m = std::is_same<ElementALayout, cutlass::layout::ColumnMajor>::value ?
      *reinterpret_cast<int64_t*>(params[3 + GRank - 1]) :
      *reinterpret_cast<int64_t*>(params[3 + GRank - 2]);
  int64_t k = std::is_same<ElementALayout, cutlass::layout::ColumnMajor>::value ?
      *reinterpret_cast<int64_t*>(params[3 + GRank - 2]) :
      *reinterpret_cast<int64_t*>(params[3 + GRank - 1]);
  int64_t n = std::is_same<ElementBLayout, cutlass::layout::ColumnMajor>::value ?
      *reinterpret_cast<int64_t*>(params[3 + size_struct + GRank - 2]) :
      *reinterpret_cast<int64_t*>(params[3 + size_struct + GRank - 1]);

  ElementAType* a = *reinterpret_cast<ElementAType**>(params[1 + param_permute[0] * size_struct]);
  ElementBType* b = *reinterpret_cast<ElementBType**>(params[1 + param_permute[1] * size_struct]);
  ElementOutputType* d =
      *reinterpret_cast<ElementOutputType**>(params[1 + param_permute[2] * size_struct]);

  SpecializedGemmFusion specialization(stream, batch_size, m, n, k, a, b, d);
  return specialization.run();
};

)bladedisc_rstr"